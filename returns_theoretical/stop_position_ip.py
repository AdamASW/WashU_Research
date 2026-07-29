import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except Exception:  # pragma: no cover
    Bounds = None
    LinearConstraint = None
    milp = None
    coo_matrix = None

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


def _solve_first_trigger_stop_ip_scipy(
    examples: List[SessionExample],
    epsilon: float,
    big_m: float,
    w_min: float,
    w_max: float,
) -> Tuple[np.ndarray, float, Dict[str, int]]:
    """SciPy MILP fallback for environments without Gurobi.

    Uses an equivalent first-trigger encoding with cumulative trigger variables.
    This reduces constraint growth from quadratic-in-session-length to linear,
    which is much more memory efficient on large datasets.
    """
    if milp is None or Bounds is None or LinearConstraint is None or coo_matrix is None:
        raise ImportError(
            "No MILP backend available. Install gurobipy or scipy with optimize.milp support."
        )

    p_max = max(len(ex.weights) for ex in examples)

    # Flat variable layout:
    # [r_0..r_{p_max-1}, t_(q,p) for all valid pairs, y_(q,p) for all valid pairs]
    # where y_(q,p) = OR_{k<=p} t_(q,k) (cumulative trigger state).
    t_index: Dict[Tuple[int, int], int] = {}
    y_index: Dict[Tuple[int, int], int] = {}

    next_idx = p_max
    for q, ex in enumerate(examples):
        for p in range(len(ex.weights)):
            t_index[q, p] = next_idx
            next_idx += 1

    for q, ex in enumerate(examples):
        for p in range(len(ex.weights)):
            y_index[q, p] = next_idx
            next_idx += 1

    n_vars = next_idx

    # Objective: maximize hits at true first-trigger stop.
    # If s_q is true stop: hit_q = y[q,s_q] - y[q,s_q-1] (with y[q,-1] := 0).
    # milp minimizes, so minimize -sum_q hit_q.
    c = np.zeros(n_vars, dtype=np.float64)
    for q, ex in enumerate(examples):
        s_q = ex.true_stop_idx
        c[y_index[q, s_q]] -= 1.0
        if s_q > 0:
            c[y_index[q, s_q - 1]] += 1.0

    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    integrality = np.zeros(n_vars, dtype=np.int32)

    # Bounds on r live on the cumulative-weight scale, not the single-position scale.
    lb[:p_max] = w_min
    ub[:p_max] = w_max

    # Binary bounds and integrality for t and y.
    for idx in list(t_index.values()) + list(y_index.values()):
        lb[idx] = 0.0
        ub[idx] = 1.0
        integrality[idx] = 1

    # Build A in sparse triplet form to avoid dense row allocations.
    row_idx: List[int] = []
    col_idx: List[int] = []
    data: List[float] = []
    lhs: List[float] = []
    rhs: List[float] = []
    n_rows = 0

    def add_row(coeffs: Dict[int, float], lo: float, hi: float) -> None:
        nonlocal n_rows
        for j, v in coeffs.items():
            if v != 0.0:
                row_idx.append(n_rows)
                col_idx.append(j)
                data.append(float(v))
        lhs.append(lo)
        rhs.append(hi)
        n_rows += 1

    # Monotone decreasing thresholds: r[p+1] - r[p] <= 0.
    for p in range(p_max - 1):
        add_row({p + 1: 1.0, p: -1.0}, -np.inf, 0.0)

    for q, ex in enumerate(examples):
        j_q = len(ex.weights)
        cum_weights = np.cumsum(ex.weights)

        # Force last trigger.
        add_row({t_index[q, j_q - 1]: 1.0}, 1.0, 1.0)

        for p in range(j_q):
            t_qp = t_index[q, p]
            y_qp = y_index[q, p]
            w_qp = float(cum_weights[p])

            # cumulative_w(p) - r[p] >= -M(1-t)
            # -> r[p] + M t <= M + cumulative_w(p)
            add_row({p: 1.0, t_qp: big_m}, -np.inf, big_m + w_qp)

            # cumulative_w(p) - r[p] <= -eps + M t
            # -> -r[p] - M t <= -eps - cumulative_w(p)
            add_row({p: -1.0, t_qp: -big_m}, -np.inf, -epsilon - w_qp)

            if p > 0:
                y_prev = y_index[q, p - 1]

                # y[p] >= y[p-1]
                add_row({y_qp: 1.0, y_prev: -1.0}, 0.0, np.inf)

                # y[p] >= t[p]
                add_row({y_qp: 1.0, t_qp: -1.0}, 0.0, np.inf)

                # y[p] <= y[p-1] + t[p]
                add_row({y_qp: 1.0, y_prev: -1.0, t_qp: -1.0}, -np.inf, 0.0)
            else:
                # Base case y[0] = t[0].
                add_row({y_qp: 1.0, t_qp: -1.0}, 0.0, 0.0)

    A = coo_matrix((np.array(data, dtype=np.float64), (np.array(row_idx), np.array(col_idx))), shape=(n_rows, n_vars)).tocsr()
    constraints = LinearConstraint(A, np.array(lhs, dtype=np.float64), np.array(rhs, dtype=np.float64))
    bounds = Bounds(lb, ub)

    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

    if not result.success:
        raise RuntimeError(f"SciPy MILP fallback failed: {result.message}")

    x = result.x
    thresholds = x[:p_max].astype(np.float64)

    predicted_stop_idx: Dict[str, int] = {}
    for q, ex in enumerate(examples):
        j_q = len(ex.weights)
        pred = j_q - 1
        if x[y_index[q, 0]] > 0.5:
            pred = 0
        else:
            for p in range(1, j_q):
                if x[y_index[q, p]] - x[y_index[q, p - 1]] > 0.5:
                    pred = p
                    break
        predicted_stop_idx[ex.session_id] = pred

    objective_hits = -float(result.fun)
    return thresholds, objective_hits, predicted_stop_idx


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
    if len(examples) == 0:
        raise ValueError("examples cannot be empty.")

    p_max = max(len(ex.weights) for ex in examples)
    all_weights = np.concatenate([np.cumsum(ex.weights) for ex in examples])

    w_min = float(all_weights.min())
    w_max = float(all_weights.max())

    if big_m is None:
        big_m = max(1.0, w_max - w_min + 1.0)

    if gp is not None and GRB is not None:
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
            cum_weights = np.cumsum(ex.weights)

            # Ensure at least one trigger so each session gets one stop.
            model.addConstr(t[q, j_q - 1] == 1, name=f"force_last_trigger_{q}")

            for p in range(j_q):
                cum_w_qp = float(cum_weights[p])

                # Big-M trigger linearization for t[q,p] ~ 1{sum_{k<=p} w_qk >= r[p]}.
                model.addConstr(cum_w_qp - r[p] >= -big_m * (1 - t[q, p]), name=f"trig_lb_{q}_{p}")
                model.addConstr(
                    cum_w_qp - r[p] <= -epsilon + big_m * t[q, p],
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

        objective_hits = float(model.objVal)
    else:
        thresholds, objective_hits, predicted_stop_idx = _solve_first_trigger_stop_ip_scipy(
            examples=examples,
            epsilon=epsilon,
            big_m=big_m,
            w_min=w_min,
            w_max=w_max,
        )

    correct = 0
    for ex in examples:
        pred = predicted_stop_idx[ex.session_id]
        if pred == ex.true_stop_idx:
            correct += 1

    hit_rate = correct / len(examples)

    return StopIPResult(
        thresholds=thresholds,
        objective_hits=float(objective_hits),
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
