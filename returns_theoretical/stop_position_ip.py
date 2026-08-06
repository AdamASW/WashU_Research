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
    assigned_type_idx: Dict[str, int]
    predicted_stop_idx: Dict[str, int]


def _prepare_problem_data(
    examples: List[SessionExample],
    epsilon: float,
) -> Tuple[int, List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    if len(examples) == 0:
        raise ValueError("examples cannot be empty.")

    n_pos = max(len(ex.weights) for ex in examples)

    cumulative: List[np.ndarray] = []
    for ex in examples:
        w = np.asarray(ex.weights, dtype=np.float64).reshape(-1)
        if len(w) == 0:
            raise ValueError("Each session must contain at least one position.")
        if ex.true_stop_idx < 0 or ex.true_stop_idx >= len(w):
            raise ValueError("true_stop_idx must be a valid 0-based position index.")
        cumulative.append(np.cumsum(w))

    r_lb = np.zeros(n_pos, dtype=np.float64)
    r_ub = np.zeros(n_pos, dtype=np.float64)

    for p in range(n_pos):
        vals = [cum[p] for cum in cumulative if p < len(cum)]
        if not vals:
            raise ValueError("No sessions found for at least one position index.")
        r_lb[p] = float(np.min(vals))
        r_ub[p] = float(np.max(vals))

    pre_big_m: List[np.ndarray] = []
    stop_big_m = np.zeros(len(examples), dtype=np.float64)

    for s, ex in enumerate(examples):
        t = ex.true_stop_idx
        cum_s = cumulative[s]

        if t > 0:
            pre_vals = []
            for p in range(t):
                # Tight Big-M so pre-stop constraints relax exactly when x_sk = 0.
                pre_vals.append(cum_s[p] - r_lb[p] + epsilon)
            pre_big_m.append(np.asarray(pre_vals, dtype=np.float64))
        else:
            pre_big_m.append(np.zeros(0, dtype=np.float64))

        # Tight Big-M so stop constraint relaxes exactly when x_sk = 0.
        stop_big_m[s] = r_ub[t] - cum_s[t]

    return n_pos, cumulative, r_lb, r_ub, pre_big_m, stop_big_m


def _solve_main_stop_ip_scipy(
    examples: List[SessionExample],
    n_customer_types: int,
    epsilon: float,
) -> Tuple[np.ndarray, float, Dict[str, int], Dict[str, int]]:
    if milp is None or Bounds is None or LinearConstraint is None or coo_matrix is None:
        raise ImportError(
            "No MILP backend available. Install gurobipy or scipy with optimize.milp support."
        )

    n_pos, cumulative, r_lb, r_ub, pre_big_m, stop_big_m = _prepare_problem_data(
        examples=examples,
        epsilon=epsilon,
    )

    n_sessions = len(examples)

    # Variable layout:
    # r[p,k] for p in [0, n_pos), k in [0, K)
    # x[s,k] for s in [0, n_sessions), k in [0, K)
    r_index: Dict[Tuple[int, int], int] = {}
    x_index: Dict[Tuple[int, int], int] = {}

    next_idx = 0
    for p in range(n_pos):
        for k in range(n_customer_types):
            r_index[p, k] = next_idx
            next_idx += 1

    for s in range(n_sessions):
        for k in range(n_customer_types):
            x_index[s, k] = next_idx
            next_idx += 1

    n_vars = next_idx

    c = np.zeros(n_vars, dtype=np.float64)
    for s in range(n_sessions):
        for k in range(n_customer_types):
            c[x_index[s, k]] = -1.0

    lb = np.full(n_vars, -np.inf, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    integrality = np.zeros(n_vars, dtype=np.int32)

    for p in range(n_pos):
        for k in range(n_customer_types):
            idx = r_index[p, k]
            lb[idx] = r_lb[p]
            ub[idx] = r_ub[p]

    for s in range(n_sessions):
        for k in range(n_customer_types):
            idx = x_index[s, k]
            lb[idx] = 0.0
            ub[idx] = 1.0
            integrality[idx] = 1

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

    # (1) Monotonicity: r[p,k] >= r[p+1,k].
    for k in range(n_customer_types):
        for p in range(n_pos - 1):
            add_row(
                {
                    r_index[p, k]: 1.0,
                    r_index[p + 1, k]: -1.0,
                },
                0.0,
                np.inf,
            )

    # (2), (3), (4)
    for s, ex in enumerate(examples):
        t = ex.true_stop_idx
        cum_s = cumulative[s]

        for k in range(n_customer_types):
            x_sk = x_index[s, k]

            # (2) Pre-stopping constraints for p < T_s:
            # W_sp <= r_pk - eps + M_pre(1 - x_sk)
            # <=> -r_pk + M_pre x_sk <= M_pre - W_sp - eps
            for p in range(t):
                m_pre = float(pre_big_m[s][p])
                add_row(
                    {
                        r_index[p, k]: -1.0,
                        x_sk: m_pre,
                    },
                    -np.inf,
                    m_pre - float(cum_s[p]) - epsilon,
                )

            # (3) Stopping constraint at T_s:
            # W_sT >= r_Tk - M_stop(1 - x_sk)
            # <=> r_Tk - M_stop x_sk <= W_sT + M_stop
            m_stop = float(stop_big_m[s])
            add_row(
                {
                    r_index[t, k]: 1.0,
                    x_sk: -m_stop,
                },
                -np.inf,
                float(cum_s[t]) + m_stop,
            )

        # (4) At most one type can explain each session.
        add_row(
            {x_index[s, k]: 1.0 for k in range(n_customer_types)},
            -np.inf,
            1.0,
        )

    # (7) Symmetry breaking: r[0,k] >= r[0,k+1].
    for k in range(n_customer_types - 1):
        add_row(
            {
                r_index[0, k]: 1.0,
                r_index[0, k + 1]: -1.0,
            },
            0.0,
            np.inf,
        )

    A = coo_matrix(
        (
            np.array(data, dtype=np.float64),
            (np.array(row_idx), np.array(col_idx)),
        ),
        shape=(n_rows, n_vars),
    ).tocsr()

    constraints = LinearConstraint(
        A,
        np.array(lhs, dtype=np.float64),
        np.array(rhs, dtype=np.float64),
    )
    bounds = Bounds(lb, ub)

    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"SciPy MILP failed: {result.message}")

    x_opt = result.x
    thresholds = np.zeros((n_pos, n_customer_types), dtype=np.float64)
    for p in range(n_pos):
        for k in range(n_customer_types):
            thresholds[p, k] = x_opt[r_index[p, k]]

    assigned_type_idx: Dict[str, int] = {}
    predicted_stop_idx: Dict[str, int] = {}

    for s, ex in enumerate(examples):
        chosen = -1
        for k in range(n_customer_types):
            if x_opt[x_index[s, k]] > 0.5:
                chosen = k
                break
        assigned_type_idx[ex.session_id] = chosen
        predicted_stop_idx[ex.session_id] = ex.true_stop_idx if chosen >= 0 else -1

    objective_hits = -float(result.fun)
    return thresholds, objective_hits, assigned_type_idx, predicted_stop_idx


def _solve_main_stop_ip_gurobi(
    examples: List[SessionExample],
    n_customer_types: int,
    epsilon: float,
) -> Tuple[np.ndarray, float, Dict[str, int], Dict[str, int]]:
    if gp is None or GRB is None:
        raise ImportError("gurobipy is not available")

    n_pos, cumulative, r_lb, r_ub, pre_big_m, stop_big_m = _prepare_problem_data(
        examples=examples,
        epsilon=epsilon,
    )

    n_sessions = len(examples)

    model = gp.Model("stop_position_ip_main")
    model.setParam("OutputFlag", 0)

    r = model.addVars(
        range(n_pos),
        range(n_customer_types),
        lb={(p, k): r_lb[p] for p in range(n_pos) for k in range(n_customer_types)},
        ub={(p, k): r_ub[p] for p in range(n_pos) for k in range(n_customer_types)},
        vtype=GRB.CONTINUOUS,
        name="r",
    )

    x = model.addVars(
        range(n_sessions),
        range(n_customer_types),
        vtype=GRB.BINARY,
        name="x",
    )

    # (1) Monotonicity: r[p,k] >= r[p+1,k].
    for k in range(n_customer_types):
        for p in range(n_pos - 1):
            model.addConstr(r[p, k] >= r[p + 1, k], name=f"mono_{p}_{k}")

    # (2), (3), (4)
    for s, ex in enumerate(examples):
        t = ex.true_stop_idx
        cum_s = cumulative[s]

        for k in range(n_customer_types):
            # (2) Pre-stopping constraints.
            for p in range(t):
                m_pre = float(pre_big_m[s][p])
                model.addConstr(
                    cum_s[p] <= r[p, k] - epsilon + m_pre * (1.0 - x[s, k]),
                    name=f"pre_{s}_{p}_{k}",
                )

            # (3) Stopping condition at observed stop.
            m_stop = float(stop_big_m[s])
            model.addConstr(
                cum_s[t] >= r[t, k] - m_stop * (1.0 - x[s, k]),
                name=f"stop_{s}_{k}",
            )

        # (4) At most one type can explain session s.
        model.addConstr(
            gp.quicksum(x[s, k] for k in range(n_customer_types)) <= 1,
            name=f"one_type_{s}",
        )

    # (7) Symmetry breaking.
    for k in range(n_customer_types - 1):
        model.addConstr(r[0, k] >= r[0, k + 1], name=f"sym_{k}")

    model.setObjective(
        gp.quicksum(x[s, k] for s in range(n_sessions) for k in range(n_customer_types)),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not return optimal status. Status code={model.status}")

    thresholds = np.zeros((n_pos, n_customer_types), dtype=np.float64)
    for p in range(n_pos):
        for k in range(n_customer_types):
            thresholds[p, k] = r[p, k].X

    assigned_type_idx: Dict[str, int] = {}
    predicted_stop_idx: Dict[str, int] = {}

    for s, ex in enumerate(examples):
        chosen = -1
        for k in range(n_customer_types):
            if x[s, k].X > 0.5:
                chosen = k
                break
        assigned_type_idx[ex.session_id] = chosen
        predicted_stop_idx[ex.session_id] = ex.true_stop_idx if chosen >= 0 else -1

    objective_hits = float(model.objVal)
    return thresholds, objective_hits, assigned_type_idx, predicted_stop_idx


def compute_vertical_mnl_scaler(
    train_data: pd.DataFrame,
    feature_list: List[str],
    session_col: str = "srch_id",
    position_col: str = "position",
    click_col: str = "click_bool",
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild the notebook scaler used before MNL estimation."""
    offered = []

    for _, group in train_data.groupby(session_col):
        g = group.sort_values(position_col)

        if g[click_col].sum() == 0:
            continue

        deepest_click = g.loc[g[click_col] == 1, position_col].max()
        g = g[g[position_col] <= deepest_click]

        x = g[feature_list].values.astype(np.float64)
        offered.append(x)

    if not offered:
        raise ValueError("No sessions with clicks were found; cannot compute scaler.")

    max_j = max(x.shape[0] for x in offered)
    x_pad = np.zeros((len(offered), max_j, len(feature_list)), dtype=np.float64)

    for i, x in enumerate(offered):
        j = x.shape[0]
        x_pad[i, :j, :] = x

    mean = x_pad.mean(axis=(0, 1))
    std = x_pad.std(axis=(0, 1)) + 1e-8

    return mean, std


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
    """Build session-level w_si and observed stopping labels.

    In the main formulation, w_si is the linear index h_si dot beta_hat.
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

        weights = x_scaled @ beta

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
    n_customer_types: int = 1,
) -> StopIPResult:
    """Solve the main IP formulation from AI-IP-Notes.

    Even though this keeps the legacy function name for compatibility,
    it now solves the K-type formulation with variables r_pk and x_sk.
    """
    if n_customer_types < 1:
        raise ValueError("n_customer_types must be >= 1.")

    if gp is not None and GRB is not None:
        thresholds, objective_hits, assigned_type_idx, predicted_stop_idx = _solve_main_stop_ip_gurobi(
            examples=examples,
            n_customer_types=n_customer_types,
            epsilon=epsilon,
        )
    else:
        thresholds, objective_hits, assigned_type_idx, predicted_stop_idx = _solve_main_stop_ip_scipy(
            examples=examples,
            n_customer_types=n_customer_types,
            epsilon=epsilon,
        )

    hit_rate = float(objective_hits) / float(len(examples))

    return StopIPResult(
        thresholds=thresholds,
        objective_hits=float(objective_hits),
        hit_rate=hit_rate,
        assigned_type_idx=assigned_type_idx,
        predicted_stop_idx=predicted_stop_idx,
    )


def fit_stop_ip_from_notebook_outputs(
    train_data: pd.DataFrame,
    feature_list: List[str],
    beta_hat: np.ndarray,
    n_customer_types: int = 1,
    epsilon: float = 1e-6,
    session_col: str = "srch_id",
    position_col: str = "position",
    click_col: str = "click_bool",
) -> Tuple[StopIPResult, List[SessionExample], np.ndarray, np.ndarray]:
    """Convenience pipeline for notebook usage.

    Steps:
    1) Rebuild the same scaler used in MNL estimation transformation
    2) Build per-session linear indices w_si = h_si dot beta_hat
    3) Solve the main K-type stop-position IP
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

    result = solve_first_trigger_stop_ip(
        examples=examples,
        epsilon=epsilon,
        n_customer_types=n_customer_types,
    )

    return result, examples, mean, std
