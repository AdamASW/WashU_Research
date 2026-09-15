// Dynamic-programming solver for the first-trigger stop-position model.
//
// The solver mirrors stop_position_dp.py.  It uses only the C++ standard
// library and operates on cumulative session weights.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef STOP_POSITION_DP_PYBIND
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#endif

namespace stop_position_dp {

struct SessionExample {
    std::string session_id;
    std::vector<double> weights;
    std::size_t true_stop_idx;
};

struct StopDPResult {
    std::vector<std::vector<double>> thresholds;
    double objective_hits;
    double hit_rate;
    std::unordered_map<std::string, int> assigned_type_idx;
    std::unordered_map<std::string, int> predicted_stop_idx;
};

using ProgressCallback = std::function<void(
    std::size_t position,
    std::size_t total_positions,
    std::size_t states_processed,
    std::size_t total_states,
    double elapsed_seconds,
    double best_score)>;

using State = std::vector<double>;

namespace {

struct PreparedData {
    std::size_t n_positions;
    std::vector<std::vector<double>> cumulative;
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
};

double tolerance(double lower, double upper) {
    return 1e-12 * std::max({1.0, std::abs(lower), std::abs(upper)});
}

bool same_state(const State& left, const State& right) {
    return left == right;
}

std::string state_key(const State& state) {
    std::string key;
    for (double value : state) {
        key += std::to_string(value);
        key.push_back('|');
    }
    return key;
}

PreparedData prepare_problem_data(
    const std::vector<SessionExample>& examples,
    double lower_bound,
    double upper_bound,
    bool fixed_bounds) {
    if (examples.empty()) {
        throw std::invalid_argument("examples cannot be empty.");
    }

    std::size_t n_positions = 0;
    for (const auto& example : examples) {
        if (example.weights.empty()) {
            throw std::invalid_argument(
                "Each session must contain at least one position.");
        }
        if (example.true_stop_idx >= example.weights.size()) {
            throw std::invalid_argument(
                "true_stop_idx must be a valid 0-based position index.");
        }
        for (double weight : example.weights) {
            if (!std::isfinite(weight)) {
                throw std::invalid_argument("Session weights must be finite.");
            }
        }
        n_positions = std::max(n_positions, example.weights.size());
    }

    std::vector<std::vector<double>> cumulative;
    cumulative.reserve(examples.size());
    for (const auto& example : examples) {
        std::vector<double> sums(example.weights.size());
        std::partial_sum(example.weights.begin(), example.weights.end(),
                         sums.begin());
        cumulative.push_back(std::move(sums));
    }

    std::vector<double> lower(n_positions, 0.0);
    std::vector<double> upper(n_positions, 0.0);
    if (fixed_bounds) {
        if (!std::isfinite(lower_bound) || !std::isfinite(upper_bound)) {
            throw std::invalid_argument("Fixed bounds must be finite floats.");
        }
        if (lower_bound > upper_bound) {
            throw std::invalid_argument(
                "r_lower_bound must be <= r_upper_bound.");
        }
        std::fill(lower.begin(), lower.end(), lower_bound);
        std::fill(upper.begin(), upper.end(), upper_bound);
    } else {
        for (std::size_t position = 0; position < n_positions; ++position) {
            bool found = false;
            for (const auto& sums : cumulative) {
                if (position < sums.size()) {
                    upper[position] = found
                        ? std::max(upper[position], sums[position])
                        : sums[position];
                    found = true;
                }
            }
            if (!found) {
                throw std::runtime_error(
                    "No sessions found for a position index.");
            }
        }
    }

    for (std::size_t position = 0; position < n_positions; ++position) {
        if (upper[position] < lower[position]) {
            throw std::runtime_error(
                "The supplied reservation-price bounds are infeasible.");
        }
    }
    return {n_positions, std::move(cumulative), std::move(lower),
            std::move(upper)};
}

std::vector<std::vector<double>> candidate_values(
    const std::vector<SessionExample>& examples,
    const PreparedData& data,
    double epsilon) {
    std::vector<std::vector<double>> candidates(data.n_positions);
    for (std::size_t position = 0; position < data.n_positions; ++position) {
        auto& values = candidates[position];
        values.push_back(data.lower_bound[position]);
        values.push_back(data.upper_bound[position]);
        for (std::size_t index = 0; index < examples.size(); ++index) {
            const auto& example = examples[index];
            const auto& sums = data.cumulative[index];
            if (position >= sums.size()) {
                continue;
            }
            if (example.true_stop_idx == position) {
                values.push_back(sums[position]);
            } else if (example.true_stop_idx > position) {
                values.push_back(sums[position] + epsilon);
            }
        }

        const double tol = tolerance(data.lower_bound[position],
                                     data.upper_bound[position]);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end(),
            [tol](double left, double right) {
                return std::abs(left - right) <= tol;
            }), values.end());
        values.erase(std::remove_if(values.begin(), values.end(),
            [&](double value) {
                return value < data.lower_bound[position] - tol ||
                       value > data.upper_bound[position] + tol;
            }), values.end());
        if (values.empty()) {
            throw std::runtime_error(
                "No candidate reservation prices at a position.");
        }
    }
    return candidates;
}

std::vector<State> ordered_states(const std::vector<double>& values,
                                  std::size_t n_types) {
    std::vector<State> states;
    State combination(n_types);
    std::function<void(std::size_t, std::size_t)> build =
        [&](std::size_t slot, std::size_t first) {
            if (slot == n_types) {
                State state = combination;
                std::reverse(state.begin(), state.end());
                states.push_back(std::move(state));
                return;
            }
            for (std::size_t index = first; index < values.size(); ++index) {
                combination[slot] = values[index];
                build(slot + 1, index);
            }
        };
    build(0, 0);
    return states;
}

class SuffixMaxTree {
public:
    explicit SuffixMaxTree(const std::vector<double>& values)
        : size_(values.size()), tree_(values.size() * 4),
          lazy_(values.size() * 4, 0.0) {
        if (size_ != 0) {
            build(1, 0, size_ - 1, values);
        }
    }

    void add(std::size_t first, double value) {
        if (first >= size_) {
            return;
        }
        add(1, 0, size_ - 1, first, size_ - 1, value);
    }

    std::pair<double, std::size_t> maximum(std::size_t first) {
        if (first >= size_) {
            return {-std::numeric_limits<double>::infinity(), size_};
        }
        const Entry result = query(1, 0, size_ - 1, first, size_ - 1);
        return {result.score, result.index};
    }

private:
    struct Entry {
        double score = -std::numeric_limits<double>::infinity();
        std::size_t index = 0;
    };

    std::size_t size_;
    std::vector<Entry> tree_;
    std::vector<double> lazy_;

    static Entry better(const Entry& left, const Entry& right) {
        return left.score >= right.score ? left : right;
    }

    void build(std::size_t node, std::size_t left, std::size_t right,
               const std::vector<double>& values) {
        if (left == right) {
            tree_[node] = {values[left], left};
            return;
        }
        const std::size_t middle = left + (right - left) / 2;
        build(node * 2, left, middle, values);
        build(node * 2 + 1, middle + 1, right, values);
        tree_[node] = better(tree_[node * 2], tree_[node * 2 + 1]);
    }

    void apply(std::size_t node, double value) {
        tree_[node].score += value;
        lazy_[node] += value;
    }

    void push(std::size_t node) {
        if (lazy_[node] == 0.0) {
            return;
        }
        apply(node * 2, lazy_[node]);
        apply(node * 2 + 1, lazy_[node]);
        lazy_[node] = 0.0;
    }

    void add(std::size_t node, std::size_t left, std::size_t right,
             std::size_t query_left, std::size_t query_right, double value) {
        if (query_left <= left && right <= query_right) {
            apply(node, value);
            return;
        }
        push(node);
        const std::size_t middle = left + (right - left) / 2;
        if (query_left <= middle) {
            add(node * 2, left, middle, query_left, query_right, value);
        }
        if (query_right > middle) {
            add(node * 2 + 1, middle + 1, right, query_left, query_right,
                value);
        }
        tree_[node] = better(tree_[node * 2], tree_[node * 2 + 1]);
    }

    Entry query(std::size_t node, std::size_t left, std::size_t right,
                std::size_t query_left, std::size_t query_right) {
        if (query_left <= left && right <= query_right) {
            return tree_[node];
        }
        push(node);
        const std::size_t middle = left + (right - left) / 2;
        Entry result;
        if (query_left <= middle) {
            result = better(result, query(node * 2, left, middle, query_left,
                                          query_right));
        }
        if (query_right > middle) {
            result = better(result, query(node * 2 + 1, middle + 1, right,
                                          query_left, query_right));
        }
        return result;
    }
};

class SparseOrthantMaxTree {
public:
    SparseOrthantMaxTree(
        const std::vector<std::vector<std::size_t>>& coordinates,
        const std::vector<double>& values,
        std::size_t dimensions)
        : dimensions_(dimensions), child_count_(std::size_t(1) << dimensions) {
        if (dimensions < 2 || dimensions > 3 ||
            coordinates.size() != values.size()) {
            throw std::invalid_argument(
                "SparseOrthantMaxTree requires 2 or 3 dimensions.");
        }
        std::vector<std::size_t> points(coordinates.size());
        std::iota(points.begin(), points.end(), 0);
        std::vector<std::size_t> lower(dimensions, 0);
        std::vector<std::size_t> upper(dimensions, 0);
        for (const auto& point : coordinates) {
            if (point.size() != dimensions) {
                throw std::invalid_argument("Invalid orthant point dimension.");
            }
            for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
                upper[dimension] = std::max(upper[dimension], point[dimension]);
            }
        }
        build(lower, upper, points, coordinates, values);
    }

    void add(const std::vector<std::size_t>& lower, double value) {
        add(0, lower, value);
    }

    std::pair<double, std::size_t> maximum(
        const std::vector<std::size_t>& lower) {
        return maximum(0, lower);
    }

private:
    struct Node {
        std::vector<std::size_t> lower;
        std::vector<std::size_t> upper;
        std::vector<std::size_t> children;
        double score = -std::numeric_limits<double>::infinity();
        std::size_t index = 0;
        double lazy = 0.0;
    };

    std::size_t dimensions_;
    std::size_t child_count_;
    std::vector<Node> nodes_;

    static std::size_t invalid_node() {
        return std::numeric_limits<std::size_t>::max();
    }

    static std::pair<double, std::size_t> better(
        const std::pair<double, std::size_t>& left,
        const std::pair<double, std::size_t>& right) {
        return left.first >= right.first ? left : right;
    }

    std::size_t build(
        const std::vector<std::size_t>& lower,
        const std::vector<std::size_t>& upper,
        const std::vector<std::size_t>& points,
        const std::vector<std::vector<std::size_t>>& coordinates,
        const std::vector<double>& values) {
        const std::size_t node_index = nodes_.size();
        nodes_.push_back({lower, upper,
                          std::vector<std::size_t>(child_count_, invalid_node()),
                          -std::numeric_limits<double>::infinity(), 0, 0.0});
        bool leaf = true;
        for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
            leaf = leaf && lower[dimension] == upper[dimension];
        }
        if (leaf) {
            for (std::size_t point : points) {
                if (values[point] > nodes_[node_index].score) {
                    nodes_[node_index].score = values[point];
                    nodes_[node_index].index = point;
                }
            }
            return node_index;
        }

        std::vector<std::size_t> middle(dimensions_);
        for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
            middle[dimension] =
                lower[dimension] + (upper[dimension] - lower[dimension]) / 2;
        }
        std::vector<std::vector<std::size_t>> child_points(child_count_);
        for (std::size_t point : points) {
            std::size_t child = 0;
            for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
                if (coordinates[point][dimension] > middle[dimension]) {
                    child |= std::size_t(1) << dimension;
                }
            }
            child_points[child].push_back(point);
        }
        for (std::size_t child = 0; child < child_count_; ++child) {
            if (child_points[child].empty()) {
                continue;
            }
            std::vector<std::size_t> child_lower = lower;
            std::vector<std::size_t> child_upper = upper;
            for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
                if (child & (std::size_t(1) << dimension)) {
                    child_lower[dimension] = middle[dimension] + 1;
                } else {
                    child_upper[dimension] = middle[dimension];
                }
            }
            nodes_[node_index].children[child] =
                build(child_lower, child_upper, child_points[child],
                      coordinates, values);
        }
        pull(node_index);
        return node_index;
    }

    bool outside(const Node& node,
                 const std::vector<std::size_t>& lower) const {
        for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
            if (node.upper[dimension] < lower[dimension]) {
                return true;
            }
        }
        return false;
    }

    bool covered(const Node& node,
                 const std::vector<std::size_t>& lower) const {
        for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
            if (node.lower[dimension] < lower[dimension]) {
                return false;
            }
        }
        return true;
    }

    void apply(std::size_t node_index, double value) {
        Node& node = nodes_[node_index];
        node.score += value;
        node.lazy += value;
    }

    void push(std::size_t node_index) {
        const double value = nodes_[node_index].lazy;
        if (value == 0.0) {
            return;
        }
        for (std::size_t child : nodes_[node_index].children) {
            if (child != invalid_node()) {
                apply(child, value);
            }
        }
        nodes_[node_index].lazy = 0.0;
    }

    void pull(std::size_t node_index) {
        Node& node = nodes_[node_index];
        for (std::size_t child : node.children) {
            if (child != invalid_node()) {
                const auto candidate =
                    std::make_pair(nodes_[child].score, nodes_[child].index);
                const auto current = std::make_pair(node.score, node.index);
                const auto best = better(current, candidate);
                node.score = best.first;
                node.index = best.second;
            }
        }
    }

    void add(std::size_t node_index,
             const std::vector<std::size_t>& lower, double value) {
        Node& node = nodes_[node_index];
        if (outside(node, lower)) {
            return;
        }
        if (covered(node, lower)) {
            apply(node_index, value);
            return;
        }
        push(node_index);
        for (std::size_t child : node.children) {
            if (child != invalid_node()) {
                add(child, lower, value);
            }
        }
        node.score = -std::numeric_limits<double>::infinity();
        pull(node_index);
    }

    std::pair<double, std::size_t> maximum(
        std::size_t node_index, const std::vector<std::size_t>& lower) {
        const Node& node = nodes_[node_index];
        if (outside(node, lower)) {
            return {-std::numeric_limits<double>::infinity(), node.index};
        }
        if (covered(node, lower)) {
            return {node.score, node.index};
        }
        push(node_index);
        std::pair<double, std::size_t> result = {
            -std::numeric_limits<double>::infinity(), node.index};
        for (std::size_t child : nodes_[node_index].children) {
            if (child != invalid_node()) {
                result = better(result, maximum(child, lower));
            }
        }
        return result;
    }
};

bool session_explained(const SessionExample& example,
                       const std::vector<double>& sums,
                       const State* previous,
                       const State& current,
                       double epsilon) {
    const std::size_t stop = example.true_stop_idx;
    if (stop == 0) {
        return std::any_of(current.begin(), current.end(),
                           [&](double value) { return value <= sums[0]; });
    }
    if (previous == nullptr) {
        throw std::invalid_argument(
            "A previous threshold state is required for t > 0.");
    }
    const double required_previous = sums[stop - 1] + epsilon;
    const double stop_value = sums[stop];
    for (std::size_t type = 0; type < current.size(); ++type) {
        if ((*previous)[type] >= required_previous &&
            current[type] <= stop_value) {
            return true;
        }
    }
    return false;
}

}  // namespace

StopDPResult solve_first_trigger_stop_dp(
    const std::vector<SessionExample>& examples,
    std::size_t n_customer_types = 1,
    double epsilon = 1e-6,
    const std::string& objective_type = "segmentation",
    double r_lower_bound = 0.0,
    double r_upper_bound = 0.0,
    bool fixed_bounds = false,
    ProgressCallback progress_callback = nullptr,
    std::size_t progress_interval = 10000) {
    if (n_customer_types < 1) {
        throw std::invalid_argument("n_customer_types must be >= 1.");
    }
    if (objective_type != "segmentation" && objective_type != "coverage") {
        throw std::invalid_argument(
            "objective_type must be segmentation or coverage.");
    }

    const PreparedData data = prepare_problem_data(
        examples, r_lower_bound, r_upper_bound, fixed_bounds);
    const auto candidates = candidate_values(examples, data, epsilon);
    std::vector<std::vector<State>> states_by_position;
    states_by_position.push_back(
        ordered_states(candidates[0], n_customer_types));
    for (std::size_t position = 1; position < data.n_positions; ++position) {
        // Customer labels are interchangeable.  Canonicalizing every
        // position removes the K!-fold duplicate states, not just at p=0.
        states_by_position.push_back(
            ordered_states(candidates[position], n_customer_types));
    }
    std::size_t total_states = 0;
    for (const auto& states : states_by_position) {
        total_states += states.size();
    }
    const auto start_time = std::chrono::steady_clock::now();
    std::size_t states_processed = 0;

    std::vector<std::vector<std::size_t>> sessions_at_position(data.n_positions);
    for (std::size_t index = 0; index < examples.size(); ++index) {
        sessions_at_position[examples[index].true_stop_idx].push_back(index);
    }

    std::vector<double> scores(states_by_position[0].size(), 0.0);
    auto report_progress = [&](std::size_t position, bool force) {
        if (!progress_callback ||
            (!force && (progress_interval == 0 ||
                        states_processed % progress_interval != 0))) {
            return;
        }
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        double best_score = scores.empty()
            ? -std::numeric_limits<double>::infinity()
            : *std::max_element(scores.begin(), scores.end());
        progress_callback(position, data.n_positions, states_processed,
                          total_states, elapsed, best_score);
    };
    for (std::size_t state_index = 0; state_index < scores.size(); ++state_index) {
        for (std::size_t session : sessions_at_position[0]) {
            if (session_explained(examples[session], data.cumulative[session],
                                  nullptr, states_by_position[0][state_index],
                                  epsilon)) {
                scores[state_index] += 1.0;
            }
        }
        ++states_processed;
    }
    report_progress(0, true);

    std::vector<std::vector<int>> backpointers;
    backpointers.push_back(std::vector<int>(scores.size(), -1));
    for (std::size_t position = 1; position < data.n_positions; ++position) {
        const auto& previous_states = states_by_position[position - 1];
        const auto& current_states = states_by_position[position];
        std::vector<double> next_scores(current_states.size(),
                                        -std::numeric_limits<double>::infinity());
        std::vector<int> next_back(current_states.size(), -1);

        if (n_customer_types == 1) {
            // Sweep current thresholds from high to low.  Once a session is
            // active (S >= b), it contributes one to every predecessor
            // suffix a >= Q.  The segment tree therefore maintains
            // V_{p-1}(a) plus all currently active hits and answers the
            // required suffix maximum, including its argmax.
            std::vector<std::size_t> session_order =
                sessions_at_position[position];
            std::sort(session_order.begin(), session_order.end(),
                      [&](std::size_t left, std::size_t right) {
                          return data.cumulative[left][position] >
                                 data.cumulative[right][position];
                      });
            std::vector<double> previous_values;
            previous_values.reserve(previous_states.size());
            for (auto state = previous_states.rbegin();
                 state != previous_states.rend(); ++state) {
                previous_values.push_back((*state)[0]);
            }
            std::vector<double> reversed_scores(scores.rbegin(), scores.rend());
            SuffixMaxTree tree(reversed_scores);
            std::size_t next_session = 0;
            for (std::size_t current = 0; current < current_states.size();
                 ++current) {
                const double threshold = current_states[current][0];
                while (next_session < session_order.size() &&
                       data.cumulative[session_order[next_session]][position] >=
                           threshold) {
                    const std::size_t session = session_order[next_session++];
                    const double required =
                        data.cumulative[session][position - 1] + epsilon;
                    const auto first = static_cast<std::size_t>(
                        std::lower_bound(previous_values.begin(),
                                         previous_values.end(), required) -
                        previous_values.begin());
                    tree.add(first, 1.0);
                }
                const auto first = static_cast<std::size_t>(
                    std::lower_bound(previous_values.begin(), previous_values.end(),
                                     threshold) - previous_values.begin());
                const auto best = tree.maximum(first);
                if (best.second < previous_states.size()) {
                    next_scores[current] = best.first;
                    next_back[current] = static_cast<int>(
                        previous_states.size() - 1 - best.second);
                }
                ++states_processed;
                report_progress(position, false);
            }
        } else if (n_customer_types == 2 || n_customer_types == 3) {
            std::vector<std::vector<std::size_t>> coordinates;
            coordinates.reserve(previous_states.size());
            for (const auto& state : previous_states) {
                std::vector<std::size_t> coordinate(n_customer_types);
                for (std::size_t type = 0; type < n_customer_types; ++type) {
                    coordinate[type] = static_cast<std::size_t>(
                        std::lower_bound(candidates[position - 1].begin(),
                                         candidates[position - 1].end(),
                                         state[type]) -
                        candidates[position - 1].begin());
                }
                coordinates.push_back(std::move(coordinate));
            }
            SparseOrthantMaxTree tree(coordinates, scores, n_customer_types);
            for (std::size_t current = 0; current < current_states.size();
                 ++current) {
                std::vector<std::size_t> current_coordinate(n_customer_types);
                for (std::size_t type = 0; type < n_customer_types; ++type) {
                    current_coordinate[type] = static_cast<std::size_t>(
                        std::lower_bound(candidates[position - 1].begin(),
                                         candidates[position - 1].end(),
                                         current_states[current][type]) -
                        candidates[position - 1].begin());
                }
                std::vector<std::pair<std::vector<std::size_t>, double>> updates;
                for (std::size_t session : sessions_at_position[position]) {
                    const double stop_value = data.cumulative[session][position];
                    const double required =
                        data.cumulative[session][position - 1] + epsilon;
                    // Canonical states are descending.  The active current
                    // coordinates therefore form a suffix, while predecessor
                    // coordinates meeting `required` form a prefix.  The
                    // existential same-type reward is true iff those regions
                    // overlap, which is equivalent to a single lower bound
                    // on the first active current coordinate.
                    std::size_t first_active = n_customer_types;
                    while (first_active > 0 &&
                           current_states[current][first_active - 1] <=
                               stop_value) {
                        --first_active;
                    }
                    if (first_active == n_customer_types) {
                        continue;
                    }
                    std::vector<std::size_t> lower(n_customer_types, 0);
                    lower[first_active] = static_cast<std::size_t>(
                        std::lower_bound(
                            candidates[position - 1].begin(),
                            candidates[position - 1].end(), required) -
                        candidates[position - 1].begin());
                    tree.add(lower, 1.0);
                    updates.emplace_back(std::move(lower), 1.0);
                }
                const auto best = tree.maximum(current_coordinate);
                for (const auto& update : updates) {
                    tree.add(update.first, -update.second);
                }
                bool valid_best = best.second < previous_states.size();
                if (valid_best) {
                    for (std::size_t type = 0; type < n_customer_types; ++type) {
                        if (previous_states[best.second][type] <
                            current_states[current][type]) {
                            valid_best = false;
                            break;
                        }
                    }
                }
                if (valid_best) {
                    next_scores[current] = best.first;
                    next_back[current] = static_cast<int>(best.second);
                } else {
                    // Keep exact semantics if a sparse boundary query cannot
                    // identify a feasible predecessor.
                    for (std::size_t previous = 0;
                         previous < previous_states.size(); ++previous) {
                        bool monotone = true;
                        for (std::size_t type = 0; type < n_customer_types;
                             ++type) {
                            if (previous_states[previous][type] <
                                current_states[current][type]) {
                                monotone = false;
                                break;
                            }
                        }
                        if (!monotone) {
                            continue;
                        }
                        double score = scores[previous];
                        for (std::size_t session :
                             sessions_at_position[position]) {
                            if (session_explained(
                                    examples[session],
                                    data.cumulative[session],
                                    &previous_states[previous],
                                    current_states[current], epsilon)) {
                                score += 1.0;
                            }
                        }
                        if (score > next_scores[current]) {
                            next_scores[current] = score;
                            next_back[current] = static_cast<int>(previous);
                        }
                    }
                }
                ++states_processed;
                report_progress(position, false);
            }
        } else {
            for (std::size_t current = 0; current < current_states.size();
                 ++current) {
                for (std::size_t previous = 0;
                     previous < previous_states.size(); ++previous) {
                    bool monotone = true;
                    for (std::size_t type = 0; type < n_customer_types;
                         ++type) {
                        if (previous_states[previous][type] <
                            current_states[current][type]) {
                            monotone = false;
                            break;
                        }
                    }
                    if (!monotone) {
                        continue;
                    }
                    double score = scores[previous];
                    for (std::size_t session :
                         sessions_at_position[position]) {
                        if (session_explained(
                                examples[session], data.cumulative[session],
                                &previous_states[previous],
                                current_states[current], epsilon)) {
                            score += 1.0;
                        }
                    }
                    if (score > next_scores[current]) {
                        next_scores[current] = score;
                        next_back[current] = static_cast<int>(previous);
                    }
                }
                ++states_processed;
                report_progress(position, false);
            }
        }
        if (progress_callback) {
            const double layer_best = next_scores.empty()
                ? -std::numeric_limits<double>::infinity()
                : *std::max_element(next_scores.begin(), next_scores.end());
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
            progress_callback(position, data.n_positions, states_processed,
                              total_states, elapsed, layer_best);
        }
        if (std::none_of(next_back.begin(), next_back.end(),
                         [](int index) { return index >= 0; })) {
            throw std::runtime_error(
                "No monotone threshold path is feasible under the supplied bounds.");
        }
        scores = std::move(next_scores);
        backpointers.push_back(std::move(next_back));
    }

    const auto best = std::max_element(scores.begin(), scores.end());
    const std::size_t final_index = static_cast<std::size_t>(
        std::distance(scores.begin(), best));
    std::vector<std::vector<double>> thresholds(data.n_positions);
    std::size_t state_index = final_index;
    for (std::size_t position = data.n_positions; position-- > 0;) {
        thresholds[position] = states_by_position[position][state_index];
        if (position > 0) {
            state_index = static_cast<std::size_t>(backpointers[position][state_index]);
        }
    }

    StopDPResult result{thresholds, *best,
                        *best / static_cast<double>(examples.size()), {}, {}};
    for (std::size_t index = 0; index < examples.size(); ++index) {
        const auto& example = examples[index];
        const std::size_t stop = example.true_stop_idx;
        const State* previous = stop == 0 ? nullptr : &thresholds[stop - 1];
        int assigned = -1;
        const State& current = thresholds[stop];
        const double stop_value = data.cumulative[index][stop];
        const double required_previous = stop == 0
            ? 0.0 : data.cumulative[index][stop - 1] + epsilon;
        for (std::size_t type = 0; type < n_customer_types; ++type) {
            const bool explains = stop == 0
                ? current[type] <= stop_value
                : (*previous)[type] >= required_previous &&
                  current[type] <= stop_value;
            if (explains) {
                assigned = static_cast<int>(type);
                break;
            }
        }
        result.assigned_type_idx[example.session_id] = assigned;
        result.predicted_stop_idx[example.session_id] = assigned >= 0
            ? static_cast<int>(stop) : -1;
    }
    return result;
}

}  // namespace stop_position_dp

#ifdef STOP_POSITION_DP_PYBIND
namespace py = pybind11;

PYBIND11_MODULE(stop_position_dp_cpp, module) {
    module.doc() = "Pybind11 bindings for the stop-position dynamic program.";
    module.def(
        "solve_first_trigger_stop_dp",
        [](const py::list& python_examples,
           std::size_t n_customer_types,
           double epsilon,
           const std::string& objective_type,
           py::object r_lower_bound,
           py::object r_upper_bound,
           py::object progress_callback,
           std::size_t progress_interval) {
            if (r_lower_bound.is_none() != r_upper_bound.is_none()) {
                throw std::invalid_argument(
                    "Provide both r_lower_bound and r_upper_bound, or neither.");
            }

            std::vector<stop_position_dp::SessionExample> examples;
            examples.reserve(python_examples.size());
            for (const py::handle item : python_examples) {
                const py::dict example = py::cast<py::dict>(item);
                stop_position_dp::SessionExample converted;
                converted.session_id =
                    py::cast<std::string>(example[py::str("session_id")]);
                converted.weights =
                    py::cast<std::vector<double>>(example[py::str("weights")]);
                converted.true_stop_idx =
                    py::cast<std::size_t>(example[py::str("true_stop_idx")]);
                examples.push_back(std::move(converted));
            }

            const bool fixed_bounds = !r_lower_bound.is_none();
            const double lower_bound = fixed_bounds
                ? py::cast<double>(r_lower_bound) : 0.0;
            const double upper_bound = fixed_bounds
                ? py::cast<double>(r_upper_bound) : 0.0;
            stop_position_dp::ProgressCallback callback = nullptr;
            if (!progress_callback.is_none()) {
                const py::function python_callback =
                    py::cast<py::function>(progress_callback);
                callback = [python_callback](
                    std::size_t position,
                    std::size_t total_positions,
                    std::size_t states_processed,
                    std::size_t total_states,
                    double elapsed_seconds,
                    double best_score) {
                    python_callback(position, total_positions, states_processed,
                                    total_states, elapsed_seconds, best_score);
                };
            }
            const auto result = stop_position_dp::solve_first_trigger_stop_dp(
                examples, n_customer_types, epsilon, objective_type,
                lower_bound, upper_bound, fixed_bounds, callback,
                progress_interval);

            py::dict assigned_type_idx;
            py::dict predicted_stop_idx;
            for (const auto& item : result.assigned_type_idx) {
                assigned_type_idx[py::str(item.first)] = item.second;
            }
            for (const auto& item : result.predicted_stop_idx) {
                predicted_stop_idx[py::str(item.first)] = item.second;
            }

            py::dict output;
            output["thresholds"] = result.thresholds;
            output["objective_hits"] = result.objective_hits;
            output["hit_rate"] = result.hit_rate;
            output["assigned_type_idx"] = assigned_type_idx;
            output["predicted_stop_idx"] = predicted_stop_idx;
            return output;
        },
        py::arg("examples"),
        py::arg("n_customer_types") = 1,
        py::arg("epsilon") = 1e-6,
        py::arg("objective_type") = "segmentation",
        py::arg("r_lower_bound") = py::none(),
        py::arg("r_upper_bound") = py::none(),
        py::arg("progress_callback") = py::none(),
        py::arg("progress_interval") = 10000);
}
#endif

#ifdef STOP_POSITION_DP_MAIN
int main() {
    using namespace stop_position_dp;
    std::size_t n_examples = 0;
    std::size_t n_types = 0;
    double epsilon = 0.0;
    if (!(std::cin >> n_examples >> n_types >> epsilon)) {
        std::cerr << "Input: N_EXAMPLES N_TYPES EPSILON, then ID N_WEIGHTS "
                     "STOP_INDEX WEIGHT...\n";
        return 1;
    }
    std::vector<SessionExample> examples;
    for (std::size_t index = 0; index < n_examples; ++index) {
        SessionExample example;
        std::size_t n_weights = 0;
        if (!(std::cin >> example.session_id >> n_weights >> example.true_stop_idx)) {
            return 1;
        }
        example.weights.resize(n_weights);
        for (double& weight : example.weights) {
            std::cin >> weight;
        }
        examples.push_back(std::move(example));
    }
    try {
        const ProgressCallback progress_callback =
            [](std::size_t position,
               std::size_t total_positions,
               std::size_t states_processed,
               std::size_t total_states,
               double elapsed_seconds,
               double best_score) {
                const double progress = total_states == 0
                    ? 100.0
                    : 100.0 * static_cast<double>(states_processed) /
                      static_cast<double>(total_states);
                std::cerr << "\rPosition " << position << "/"
                          << total_positions << " | Progress " << progress
                          << "% | States " << states_processed << "/"
                          << total_states << " | Elapsed "
                          << elapsed_seconds << " s | Best score "
                          << best_score << std::flush;
                if (position + 1 == total_positions &&
                    states_processed == total_states) {
                    std::cerr << '\n';
                }
            };
        const auto result = solve_first_trigger_stop_dp(
            examples, n_types, epsilon, "segmentation", 0.0, 0.0, false,
            progress_callback);
        std::cout << std::setprecision(17) << "objective_hits "
                  << result.objective_hits << "\nhit_rate "
                  << result.hit_rate << "\nthresholds\n";
        for (const auto& row : result.thresholds) {
            for (double value : row) {
                std::cout << value << ' ';
            }
            std::cout << '\n';
        }
        for (const auto& example : examples) {
            std::cout << example.session_id << " assigned_type "
                      << result.assigned_type_idx.at(example.session_id)
                      << " predicted_stop "
                      << result.predicted_stop_idx.at(example.session_id)
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif