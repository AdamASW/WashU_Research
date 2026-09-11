"""Dynamic-programming solver for the first-trigger stop-position model.

The public solver arguments and result type mirror ``stop_position_ip``.  The
IP's binary assignment variables are eliminated: each DP transition rewards
sessions whose observed stop is explained by at least one customer type.
"""

from itertools import combinations_with_replacement, product
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .stop_position_ip import (
        SessionExample,
        StopIPResult,
        build_session_examples_from_mnl,
        compute_vertical_mnl_scaler,
    )
except ImportError:  # Supports importing this file from its directory.
    from stop_position_ip import (
        SessionExample,
        StopIPResult,
        build_session_examples_from_mnl,
        compute_vertical_mnl_scaler,
    )


ThresholdState = Tuple[float, ...]


def _prepare_problem_data(
    examples: List[SessionExample],
    epsilon: float,
    r_lower_bound: Optional[float],
    r_upper_bound: Optional[float],
) -> Tuple[int, List[np.ndarray], np.ndarray, np.ndarray]:
    """Build cumulative weights and the same position bounds as the IP."""
    if not examples:
        raise ValueError("examples cannot be empty.")

    n_pos = max(len(np.asarray(ex.weights).reshape(-1)) for ex in examples)
    cumulative: List[np.ndarray] = []

    for ex in examples:
        weights = np.asarray(ex.weights, dtype=np.float64).reshape(-1)
        if len(weights) == 0:
            raise ValueError("Each session must contain at least one position.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("Session weights must be finite.")
        if ex.true_stop_idx < 0 or ex.true_stop_idx >= len(weights):
            raise ValueError("true_stop_idx must be a valid 0-based position index.")
        cumulative.append(np.cumsum(weights))

    if (r_lower_bound is None) != (r_upper_bound is None):
        raise ValueError("Provide both r_lower_bound and r_upper_bound, or neither.")

    if r_lower_bound is None:
        # This is the bound construction used by stop_position_ip.py.
        r_lb = np.zeros(n_pos, dtype=np.float64)
        r_ub = np.zeros(n_pos, dtype=np.float64)
        for p in range(n_pos):
            values = [cum[p] for cum in cumulative if p < len(cum)]
            if not values:
                raise ValueError(f"No sessions found for position {p}.")
            r_ub[p] = float(np.max(values))
    else:
        if not np.isfinite(r_lower_bound) or not np.isfinite(r_upper_bound):
            raise ValueError("Fixed bounds must be finite floats.")
        if float(r_lower_bound) > float(r_upper_bound):
            raise ValueError("r_lower_bound must be <= r_upper_bound.")
        r_lb = np.full(n_pos, float(r_lower_bound), dtype=np.float64)
        r_ub = np.full(n_pos, float(r_upper_bound), dtype=np.float64)

    if np.any(r_ub < r_lb):
        raise RuntimeError("The supplied reservation-price bounds are infeasible.")

    return n_pos, cumulative, r_lb, r_ub


def _candidate_values(
    examples: List[SessionExample],
    cumulative: List[np.ndarray],
    r_lb: np.ndarray,
    r_ub: np.ndarray,
    epsilon: float,
) -> List[np.ndarray]:
    """Construct the finite candidate set R_p from DP-Notes.md."""
    candidates: List[np.ndarray] = []

    for p in range(len(r_lb)):
        values = {float(r_lb[p]), float(r_ub[p])}
        for ex, cum in zip(examples, cumulative):
            if p >= len(cum):
                continue
            if ex.true_stop_idx == p:
                values.add(float(cum[p]))
            elif ex.true_stop_idx > p:
                values.add(float(cum[p] + epsilon))

        # Values outside the IP bounds cannot be selected.  The small tolerance
        # only protects against roundoff at an explicitly supplied endpoint.
        tol = 1e-12 * max(1.0, abs(r_lb[p]), abs(r_ub[p]))
        valid = sorted(
            value for value in values
            if r_lb[p] - tol <= value <= r_ub[p] + tol
        )
        if not valid:
            raise RuntimeError(f"No candidate reservation prices at position {p}.")
        candidates.append(np.asarray(valid, dtype=np.float64))

    return candidates


def _ordered_states(values: np.ndarray, n_customer_types: int) -> List[ThresholdState]:
    """Return position states, applying IP symmetry breaking at position 0."""
    # combinations_with_replacement avoids constructing the discarded
    # permutations of the position-0 symmetry-broken states.
    return [
        tuple(float(value) for value in reversed(combination))
        for combination in combinations_with_replacement(
            values, n_customer_types
        )
    ]


def _session_explained(
    ex: SessionExample,
    cumulative: np.ndarray,
    previous: Optional[ThresholdState],
    current: ThresholdState,
    epsilon: float,
) -> bool:
    """Test the IP pre-stop and stop constraints for all customer types."""
    t = ex.true_stop_idx
    if t == 0:
        return any(current[k] <= cumulative[0] for k in range(len(current)))
    if previous is None:
        raise ValueError("A previous threshold state is required for t > 0.")
    required_previous = cumulative[t - 1] + epsilon
    stop_value = cumulative[t]
    return any(
        previous[k] >= required_previous and current[k] <= stop_value
        for k in range(len(current))
    )


def _base_rewards(
    states: List[ThresholdState],
    sessions: List[Tuple[SessionExample, np.ndarray]],
) -> np.ndarray:
    """Vectorized rewards for position zero states."""
    state_values = np.asarray(states, dtype=np.float64)
    rewards = np.zeros(len(states), dtype=np.int64)
    for _, cumulative in sessions:
        rewards += np.any(state_values <= cumulative[0], axis=1)
    return rewards


def _best_transition_scores(
    previous_states: np.ndarray,
    current_states: np.ndarray,
    previous_scores: np.ndarray,
    sessions: List[Tuple[SessionExample, np.ndarray]],
    epsilon: float,
    current_chunk_size: int = 32,
    previous_chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find best predecessor scores without materializing all transitions."""
    n_current = len(current_states)
    n_previous = len(previous_states)
    best_scores = np.full(n_current, -np.inf, dtype=np.float64)
    best_previous_indices = np.full(n_current, -1, dtype=np.int64)

    for current_start in range(0, n_current, current_chunk_size):
        current_stop = min(current_start + current_chunk_size, n_current)
        current_chunk = current_states[current_start:current_stop]
        chunk_best_scores = np.full(current_stop - current_start, -np.inf)
        chunk_best_indices = np.full(current_stop - current_start, -1, dtype=np.int64)

        for previous_start in range(0, n_previous, previous_chunk_size):
            previous_stop = min(previous_start + previous_chunk_size, n_previous)
            previous_chunk = previous_states[previous_start:previous_stop]
            previous_score_chunk = previous_scores[previous_start:previous_stop]
            feasible = np.all(
                previous_chunk[None, :, :] >= current_chunk[:, None, :], axis=2
            )
            rewards = np.zeros(
                (current_stop - current_start, previous_stop - previous_start),
                dtype=np.int32,
            )
            for example, cumulative in sessions:
                stop_index = example.true_stop_idx
                required_previous = cumulative[stop_index - 1] + epsilon
                stop_value = cumulative[stop_index]
                type_matches = np.logical_and(
                    previous_chunk[None, :, :] >= required_previous,
                    current_chunk[:, None, :] <= stop_value,
                )
                rewards += np.any(type_matches, axis=2)

            tile_scores = np.where(
                feasible,
                previous_score_chunk[None, :] + rewards,
                -np.inf,
            )
            tile_indices = np.argmax(tile_scores, axis=1)
            tile_best_scores = tile_scores[
                np.arange(current_stop - current_start), tile_indices
            ]
            update = tile_best_scores > chunk_best_scores
            chunk_best_scores[update] = tile_best_scores[update]
            chunk_best_indices[update] = previous_start + tile_indices[update]

        best_scores[current_start:current_stop] = chunk_best_scores
        best_previous_indices[current_start:current_stop] = chunk_best_indices

    return best_scores, best_previous_indices


def _solve_dp(
    examples: List[SessionExample],
    n_customer_types: int,
    epsilon: float,
    r_lower_bound: Optional[float],
    r_upper_bound: Optional[float],
) -> Tuple[np.ndarray, float, Dict[str, int], Dict[str, int]]:
    n_pos, cumulative, r_lb, r_ub = _prepare_problem_data(
        examples, epsilon, r_lower_bound, r_upper_bound
    )
    candidates = _candidate_values(examples, cumulative, r_lb, r_ub, epsilon)

    # Position 0 carries the IP's symmetry break r[0,k] >= r[0,k+1].
    states_by_position = [_ordered_states(candidates[0], n_customer_types)]
    states_by_position.extend(
        list(
            tuple(float(x) for x in state)
            for state in product(values, repeat=n_customer_types)
        )
        for values in candidates[1:]
    )

    sessions_at_position: List[List[Tuple[SessionExample, np.ndarray]]] = [
        [] for _ in range(n_pos)
    ]
    for ex, cum in zip(examples, cumulative):
        sessions_at_position[ex.true_stop_idx].append((ex, cum))

    first_states = states_by_position[0]
    first_rewards = _base_rewards(first_states, sessions_at_position[0])
    dp: Dict[ThresholdState, float] = {
        state: float(score)
        for state, score in zip(first_states, first_rewards)
    }
    backpointers: List[Dict[ThresholdState, Optional[ThresholdState]]] = [
        {state: None for state in dp}
    ]

    for p in range(1, n_pos):
        previous_states = list(dp)
        current_states = states_by_position[p]
        previous_array = np.asarray(previous_states, dtype=np.float64)
        current_array = np.asarray(current_states, dtype=np.float64)
        previous_scores = np.asarray(
            [dp[state] for state in previous_states], dtype=np.float64
        )

        best_scores, best_previous_indices = _best_transition_scores(
            previous_states=previous_array,
            current_states=current_array,
            previous_scores=previous_scores,
            sessions=sessions_at_position[p],
            epsilon=epsilon,
        )

        next_dp: Dict[ThresholdState, float] = {}
        next_back: Dict[ThresholdState, Optional[ThresholdState]] = {}
        for current_index, current in enumerate(current_states):
            if not np.isfinite(best_scores[current_index]):
                continue
            previous = previous_states[best_previous_indices[current_index]]
            next_dp[current] = float(best_scores[current_index])
            next_back[current] = previous

        if not next_dp:
            raise RuntimeError(
                "No monotone threshold path is feasible under the supplied bounds."
            )
        dp = next_dp
        backpointers.append(next_back)

    final_state = max(dp, key=dp.get)
    path = [final_state]
    for p in range(n_pos - 1, 0, -1):
        previous = backpointers[p][path[-1]]
        if previous is None:
            raise RuntimeError("Failed to reconstruct the DP threshold path.")
        path.append(previous)
    path.reverse()
    thresholds = np.asarray(path, dtype=np.float64)

    assigned_type_idx: Dict[str, int] = {}
    predicted_stop_idx: Dict[str, int] = {}
    for ex, cum in zip(examples, cumulative):
        t = ex.true_stop_idx
        previous = None if t == 0 else tuple(thresholds[t - 1])
        current = tuple(thresholds[t])
        assigned = -1
        for k in range(n_customer_types):
            if t == 0:
                explains = current[k] <= cum[t]
            else:
                explains = (
                    previous[k] >= cum[t - 1] + epsilon
                    and current[k] <= cum[t]
                )
            if explains:
                assigned = k
                break
        assigned_type_idx[ex.session_id] = assigned
        predicted_stop_idx[ex.session_id] = t if assigned >= 0 else -1

    return thresholds, float(dp[final_state]), assigned_type_idx, predicted_stop_idx


def solve_first_trigger_stop_dp(
    examples: List[SessionExample],
    epsilon: float = 1e-6,
    n_customer_types: int = 1,
    objective_type: str = "segmentation",
    r_lower_bound: float = None,
    r_upper_bound: float = None,
) -> StopIPResult:
    """Solve the stop-position model with dynamic programming."""
    if n_customer_types < 1:
        raise ValueError("n_customer_types must be >= 1.")
    if objective_type not in {"segmentation", "coverage"}:
        raise ValueError("objective_type must be one of {'segmentation', 'coverage'}.")

    thresholds, objective_hits, assigned, predicted = _solve_dp(
        examples=examples,
        n_customer_types=n_customer_types,
        epsilon=epsilon,
        r_lower_bound=r_lower_bound,
        r_upper_bound=r_upper_bound,
    )
    return StopIPResult(
        thresholds=thresholds,
        objective_hits=objective_hits,
        hit_rate=objective_hits / float(len(examples)),
        assigned_type_idx=assigned,
        predicted_stop_idx=predicted,
    )


def fit_stop_dp_from_notebook_outputs(
    train_data,
    feature_list: List[str],
    beta_hat: np.ndarray,
    n_customer_types: int = 1,
    epsilon: float = 1e-6,
    objective_type: str = "segmentation",
    r_lower_bound: float = None,
    r_upper_bound: float = None,
    session_col: str = "srch_id",
    position_col: str = "position",
    click_col: str = "click_bool",
):
    """Notebook-compatible wrapper with the IP solver's input signature."""
    mean, std = compute_vertical_mnl_scaler(
        train_data, feature_list, session_col, position_col, click_col
    )
    examples = build_session_examples_from_mnl(
        train_data, feature_list, beta_hat, mean, std,
        session_col, position_col, click_col
    )
    result = solve_first_trigger_stop_dp(
        examples, epsilon, n_customer_types, objective_type,
        r_lower_bound, r_upper_bound
    )
    return result, examples, mean, std
