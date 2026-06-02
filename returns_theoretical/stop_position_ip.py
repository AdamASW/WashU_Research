import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


@dataclass
class SessionExample:
    session_id: str
    weights: np.ndarray
    true_stop_idx: int


@dataclass
class StopIPResult:
    thresholds: np.ndarray
    objective_hits: float
    hit_rate: float
    predicted_stop_idx: Dict[str, int]


def compute_vertical_mnl_scaler(
    train_data: pd.DataFrame,
    feature_list: List[str],
    session_col: str = "srch_id",
    position_col: str = "position",
    click_col: str = "click_bool",
) -> Tuple[np.ndarray, np.ndarray]:
    """Replicates the notebook's mean/std logic used before MNL estimation.

    The scaler is computed on padded arrays after truncating each session to its deepest click,
    matching the data transformation in vertical_diff_model.ipynb.
    """
    offered = []

    for session_id, group in train_data.groupby(session_col):
        _ = session_id
        g = group.sort_values(position_col)

        if g[click_col].sum() == 0:
            continue

        deepest_click = g.loc[g[click_col] == 1, position_col].max()
        g = g[g[position_col] <= deepest_click]

        x = g[feature_list].values.astype(np.float32)
        offered.append(x)

    if not offered:
        raise ValueError("No sessions with clicks were found; cannot compute scaler.")

    max_j = max(x.shape[0] for x in offered)
    x_pad = np.zeros((len(offered), max_j, len(feature_list)), dtype=np.float32)

    for i, x in enumerate(offered):
        j = x.shape[0]
        x_pad[i, :j, :] = x

    mean = x_pad.mean(axis=(0, 1))
    std = x_pad.std(axis=(0, 1)) + 1e-8

    return mean.astype(np.float64), std.astype(np.float64)


def build_session_examples_from_mnl(
    data: pd.DataFrame,
    feature_list: List[str],
    beta_hat: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    session_col: str = "srch_id",
    position_col: str = "position",
    click_col: str = "click_bool",
) -> List[SessionExample]:
    """Builds session-level MNL weights and observed stopping labels.

    True stop index is the deepest clicked position after sorting by rank position.
    Returns 0-based position indices in each truncated session.
    """
    beta = np.asarray(beta_hat, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)

    if len(beta) != len(feature_list):
        raise ValueError("beta_hat length must match feature_list length.")

    examples: List[SessionExample] = []

    for session_id, group in data.groupby(session_col):
        g = group.sort_values(position_col).copy()

        if g[click_col].sum() == 0:
            continue

        deepest_click_rank = g.loc[g[click_col] == 1, position_col].max()
        g = g[g[position_col] <= deepest_click_rank].copy()

        x_raw = g[feature_list].values.astype(np.float64)
        x_scaled = (x_raw - mean.reshape(1, -1)) / std.reshape(1, -1)

        utilities = x_scaled @ beta
        weights = np.exp(utilities)

        # deepest click is the observed stop by design
        local_stop = int(np.where(g[position_col].values == deepest_click_rank)[0][0])

        examples.append(
            SessionExample(
                session_id=str(session_id),
                weights=weights,
                true_stop_idx=local_stop,
            )
        )

    if not examples:
        raise ValueError("No valid session examples were built.")

    return examples


def solve_first_trigger_stop_ip(
    examples: List[SessionExample],
    epsilon: float = 1e-6,
    big_m: float = None,
) -> StopIPResult:
    """First IP prototype with decreasing position thresholds.

    Decision variables:
    - r_p: reservation threshold by position p
    - t_{q,p}: trigger if session q, position p has weight above threshold
    - z_{q,p}: predicted stop position (first triggered position)

    Objective maximizes correctly predicted stop positions.
    """
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required for solve_first_trigger_stop_ip.")

    if len(examples) == 0:
        raise ValueError("examples cannot be empty.")

    p_max = max(len(ex.weights) for ex in examples)
    all_weights = np.concatenate([ex.weights for ex in examples])

    w_min = float(all_weights.min())
    w_max = float(all_weights.max())

    if big_m is None:
        big_m = max(1.0, w_max - w_min + 1.0)

    model = gp.Model("stop_position_ip")
    model.setParam("OutputFlag", 0)

    r = model.addVars(range(p_max), lb=w_min, ub=w_max, vtype=GRB.CONTINUOUS, name="r")

    t = {}
    z = {}

    for q, ex in enumerate(examples):
        j_q = len(ex.weights)
        for p in range(j_q):
            t[q, p] = model.addVar(vtype=GRB.BINARY, name=f"t_{q}_{p}")
            z[q, p] = model.addVar(vtype=GRB.BINARY, name=f"z_{q}_{p}")

    # Monotone decreasing reservation thresholds by position.
    for p in range(p_max - 1):
        model.addConstr(r[p + 1] <= r[p], name=f"mono_{p}")

    for q, ex in enumerate(examples):
        j_q = len(ex.weights)

        # Ensure at least one trigger so each session gets one stop.
        model.addConstr(t[q, j_q - 1] == 1, name=f"force_last_trigger_{q}")

        for p in range(j_q):
            w_qp = float(ex.weights[p])

            # Big-M trigger linearization for t[q,p] ~ 1{w_qp >= r[p]}.
            model.addConstr(w_qp - r[p] >= -big_m * (1 - t[q, p]), name=f"trig_lb_{q}_{p}")
            model.addConstr(
                w_qp - r[p] <= -epsilon + big_m * t[q, p],
                name=f"trig_ub_{q}_{p}",
            )

            # Stop can only happen at triggered positions.
            model.addConstr(z[q, p] <= t[q, p], name=f"z_le_t_{q}_{p}")

            # First-trigger logic.
            if p > 0:
                prev_sum = gp.quicksum(t[q, k] for k in range(p))
                for k in range(p):
                    model.addConstr(z[q, p] <= 1 - t[q, k], name=f"no_prev_{q}_{p}_{k}")
                model.addConstr(z[q, p] >= t[q, p] - prev_sum, name=f"first_lb_{q}_{p}")
            else:
                model.addConstr(z[q, p] >= t[q, p], name=f"first_lb_{q}_{p}")

        # Exactly one predicted stop per session.
        model.addConstr(gp.quicksum(z[q, p] for p in range(j_q)) == 1, name=f"one_stop_{q}")

    # Maximize number of correctly predicted stopping points.
    objective_terms = []
    for q, ex in enumerate(examples):
        objective_terms.append(z[q, ex.true_stop_idx])

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not return optimal status. Status code={model.status}")

    thresholds = np.array([r[p].X for p in range(p_max)], dtype=np.float64)

    predicted_stop_idx: Dict[str, int] = {}
    correct = 0

    for q, ex in enumerate(examples):
        j_q = len(ex.weights)
        pred = None

        for p in range(j_q):
            if z[q, p].X > 0.5:
                pred = p
                break

        if pred is None:
            pred = j_q - 1

        predicted_stop_idx[ex.session_id] = pred
        if pred == ex.true_stop_idx:
            correct += 1

    hit_rate = correct / len(examples)

    return StopIPResult(
        thresholds=thresholds,
        objective_hits=float(model.objVal),
        hit_rate=float(hit_rate),
        predicted_stop_idx=predicted_stop_idx,
    )


def fit_stop_ip_from_notebook_outputs(
    train_data: pd.DataFrame,
    feature_list: List[str],
    beta_hat: np.ndarray,
    session_col: str = "srch_id",
    position_col: str = "position",
    click_col: str = "click_bool",
) -> Tuple[StopIPResult, List[SessionExample], np.ndarray, np.ndarray]:
    """Convenience pipeline for notebook usage.

    Steps:
    1) Rebuild the same scaler used in MNL estimation transformation
    2) Build per-session MNL weights
    3) Solve first-trigger stop IP
    """
    mean, std = compute_vertical_mnl_scaler(
        train_data=train_data,
        feature_list=feature_list,
        session_col=session_col,
        position_col=position_col,
        click_col=click_col,
    )

    examples = build_session_examples_from_mnl(
        data=train_data,
        feature_list=feature_list,
        beta_hat=beta_hat,
        mean=mean,
        std=std,
        session_col=session_col,
        position_col=position_col,
        click_col=click_col,
    )

    result = solve_first_trigger_stop_ip(examples)
    return result, examples, mean, std
