// Dynamic-programming solver for the first-trigger stop-position model.
//
// The solver mirrors stop_position_dp.py.  It uses only the C++ standard
// library and operates on cumulative session weights.

#include <algorithm>
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

std::vector<State> product_states(const std::vector<double>& values,
                                   std::size_t n_types) {
    std::vector<State> states;
    State state(n_types);
    std::function<void(std::size_t)> build = [&](std::size_t slot) {
        if (slot == n_types) {
            states.push_back(state);
            return;
        }
        for (double value : values) {
            state[slot] = value;
            build(slot + 1);
        }
    };
    build(0);
    return states;
}

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
    bool fixed_bounds = false) {
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
        states_by_position.push_back(
            product_states(candidates[position], n_customer_types));
    }

    std::vector<std::vector<std::size_t>> sessions_at_position(data.n_positions);
    for (std::size_t index = 0; index < examples.size(); ++index) {
        sessions_at_position[examples[index].true_stop_idx].push_back(index);
    }

    std::vector<double> scores(states_by_position[0].size(), 0.0);
    for (std::size_t state_index = 0; state_index < scores.size(); ++state_index) {
        for (std::size_t session : sessions_at_position[0]) {
            if (session_explained(examples[session], data.cumulative[session],
                                  nullptr, states_by_position[0][state_index],
                                  epsilon)) {
                scores[state_index] += 1.0;
            }
        }
    }

    std::vector<std::vector<int>> backpointers;
    backpointers.push_back(std::vector<int>(scores.size(), -1));
    for (std::size_t position = 1; position < data.n_positions; ++position) {
        const auto& previous_states = states_by_position[position - 1];
        const auto& current_states = states_by_position[position];
        std::vector<double> next_scores(current_states.size(),
                                        -std::numeric_limits<double>::infinity());
        std::vector<int> next_back(current_states.size(), -1);
        for (std::size_t current = 0; current < current_states.size(); ++current) {
            for (std::size_t previous = 0; previous < previous_states.size(); ++previous) {
                bool monotone = true;
                for (std::size_t type = 0; type < n_customer_types; ++type) {
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
                for (std::size_t session : sessions_at_position[position]) {
                    if (session_explained(examples[session],
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
        const auto result = solve_first_trigger_stop_dp(
            examples, n_types, epsilon);
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