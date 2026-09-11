# Dynamic-Programming Formulation

## Data and cumulative weights

Let $s \in \mathcal{S}$ index sessions and let $n_s$ denote the number of
offered hotels in session $s$:

$$
\mathcal{P}_s = \{0,1,\ldots,n_s-1\}.
$$

Let $w_{s,p}>0$ be the MNL weight of the hotel at position $p$ and define the
cumulative weight:

$$
W_{s,p} = \sum_{j=0}^{p} w_{s,j},
\qquad
W_{s,0}<W_{s,1}<\cdots<W_{s,n_s-1}.
$$

Let $T_s \in \mathcal{P}_s$ be the observed stopping position and let
$\varepsilon>0$.

## One customer type

The reservation-price sequence is

$$
r_0 \ge r_1 \ge \cdots \ge r_{P-1},
$$

where $P=\max_{s\in\mathcal{S}} n_s$. A session $s$ is explained by its
observed stop $T_s$ exactly when

$$
W_{s,p}+\varepsilon \le r_p
\quad \forall p<T_s,
\qquad
r_{T_s}\le W_{s,T_s}.
$$

### Finite candidate sets

For each position $p$, define

$$
\mathcal{R}_p
=
\{r_p^{\mathrm{LB}},r_p^{\mathrm{UB}}\}
\cup
\{W_{s,p}:T_s=p\}
\cup
\{W_{s,p}+\varepsilon:T_s>p\}.
$$

Only candidates satisfying

$$
r_p^{\mathrm{LB}}\le r_p\le r_p^{\mathrm{UB}}
$$

are retained. With no fixed bounds supplied, the implementation uses

$$
r_p^{\mathrm{LB}}=0,
\qquad
r_p^{\mathrm{UB}}=\max_{s:p<n_s} W_{s,p}.
$$

### Markov reduction

Because $W_{s,p}$ is increasing in $p$ and $r_p$ is nonincreasing, the
pre-stop constraints are equivalent to the final pre-stop constraint:

$$
r_{T_s-1}\ge W_{s,T_s-1}+\varepsilon.
$$

Define

$$
Q_s=W_{s,T_s-1}+\varepsilon,
\qquad
S_s=W_{s,T_s}.
$$

For $p>0$, the transition reward is

$$
g_p(a,b)
=
\left|\left\{
s:T_s=p,\ Q_s\le a,\ S_s\ge b
\right\}\right|.
$$

At position zero, define

$$
g_0(b)
=
\left|\left\{s:T_s=0,\ S_s\ge b\right\}\right|.
$$

### Scalar DP

Let $D_p(b)$ be the maximum number of explained sessions through position $p$
conditional on $r_p=b$:

$$
D_0(b)=g_0(b),
\qquad b\in\mathcal{R}_0,
$$

and, for $p>0$,

$$
D_p(b)=
\max_{\substack{a\in\mathcal{R}_{p-1}\\a\ge b}}
\left[D_{p-1}(a)+g_p(a,b)\right].
$$

The optimal objective is

$$
\max_{b\in\mathcal{R}_{P-1}}D_{P-1}(b).
$$

## Multiple customer types

Let $K\ge 1$ and define the threshold vector at position $p$:

$$
\mathbf{r}_p=(r_{p,0},r_{p,1},\ldots,r_{p,K-1}).
$$

Each coordinate takes values in $\mathcal{R}_p$. A DP state is therefore

$$
\mathbf{b}\in\mathcal{R}_p^K.
$$

The symmetry-breaking condition at position zero is

$$
r_{0,0}\ge r_{0,1}\ge\cdots\ge r_{0,K-1}.
$$

For $p>0$, a transition from $\mathbf{a}$ to $\mathbf{b}$ is feasible when

$$
a_k\ge b_k
\qquad \forall k\in\{0,\ldots,K-1\}.
$$

For a session with $T_s=p>0$, the transition reward is

$$
g_p(\mathbf{a},\mathbf{b})
=
\left|\left\{
s:T_s=p,\ \exists k\in\{0,\ldots,K-1\}
\text{ such that }a_k\ge Q_s\text{ and }b_k\le S_s
\right\}\right|.
$$

The type index $k$ is the same in both inequalities. For $p=0$:

$$
g_0(\mathbf{b})
=
\left|\left\{
s:T_s=0,\ \exists k\text{ such that }b_k\le S_s
\right\}\right|.
$$

Let $D_p(\mathbf{b})$ be the maximum number of explained sessions through
position $p$ conditional on $\mathbf{r}_p=\mathbf{b}$:

$$
D_0(\mathbf{b})=g_0(\mathbf{b}),
$$

and, for $p>0$,

$$
D_p(\mathbf{b})=
\max_{\substack{\mathbf{a}\in\mathcal{R}_{p-1}^K\\
\mathbf{a}\succeq\mathbf{b}}}
\left[D_{p-1}(\mathbf{a})+g_p(\mathbf{a},\mathbf{b})\right],
$$

where

$$
\mathbf{a}\succeq\mathbf{b}
\quad\Longleftrightarrow\quad
a_k\ge b_k\ \ \forall k.
$$

The optimal objective is

$$
\max_{\mathbf{b}\in\mathcal{R}_{P-1}^K}D_{P-1}(\mathbf{b}).
$$

Since each session contributes through an existential condition over types,
the DP counts a covered session once. Thus the `segmentation` and `coverage`
options have the same objective value in this DP implementation.

## Implementation notes

The number of states at position $p$ is approximately

$$
|\mathcal{R}_p|^K.
$$

The implementation avoids allocating the full transition matrix

$$
|\text{current states}|\times|\text{previous states}|.
$$

Instead, it evaluates current and previous states in NumPy tiles, computes
same-type rewards within each tile, and retains only the best predecessor and
score for each current state. At position zero, ordered states are generated
directly rather than by generating and discarding all permutations.

## Interface requirement

The DP solver must preserve the input and output interfaces of
`stop_position_ip.py`, including session examples, epsilon, customer-type
count, objective choice, optional bounds, threshold output, objective value,
hit rate, assignments, and predicted stopping positions.