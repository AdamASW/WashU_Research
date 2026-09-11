# DP Formulation Notes

Goal is to rewrite the IP solver in stop_position_ip.py as a DP in stop_position_dp.py. 

## K = 1:

### Initial Params:

- Hotels appear in sequentially in data as 1,2,...,n_s per search s.
- W_sp = \sum_{j=0}^p W_sj, W_sp < ... < W_{s,n_s - 1}.
- Session correctly expalined by position T_s if W_sp + \epsilon <= r_p \forall p < T_s and r_{T_s} <= W_{s, T_s}.
- To instead solve with an IP, we can solve with a DP by discretizing the values of the reservation prices. Specifically, we can define the set R_p of possible values for r_p as {r_p^{LB}, r_p^{UB}} \cup {W_sp : T_s = p} \cup {W_sp + \epsilon : T_s > p}.

### Positive Weight Structure Implications:

- For each s, define Q_s = W_{s, T_s - 1} + \epsilon.
- Due to the monotone increasing nature of the cumulative weights and the monotone decreasing nature of the reservation prices, all pre-stop constraints reduce to r_{T_s - 1} >= W_{s, T_s - 1} + \epsilon.
- Further, s is explained by T_s iff. r_{T_s - 1} >= Q_s and r_{T_s} <= S_s where S_s = W_s * T_s.

### Markov Structure:

- Thus, we can represent each search as a tuple s = (T_s, Q_s, S_s).
- For each p > 0, define reward function g_p(a,b) = #{s : T_s = p, Q_s < a, S_s >= b}.
- Here, a = r_{p-1} and b = r_p, which are the only parameters needed to define if a search s is explained.
- So, we have g_p(r_{p-1}, r_p) as the number of sessions explained by the choices of r_p and r_{p-1}. 
- Base case: g_0(r_0) = #{s : T_s = 0, S_s >= r_0}.
- Objective decomposes to max_r{g_0(r_0) + \sum_{p=1}^{p-1} g_p(r_{p-1}, r_p)}.

### Final DP Formulation:

Let R_p be the candidate values for r_p as defined above.

Define DP[p,b] = max # of sessions explained through p, given r_p = b.
- For every b \in R_0, DP[0,b] = g_0(b) provided r_0^{LB} <= b r_0^{UB}.
- For p > 0 : DP[p,b] = max_{a \in R_{p-1} : a >= b}[DP[p-1, a] + g_p(a,b)] for r_{p-1} >= r_p.

Finally, we have max_{b \in R_{p-1}}DP[P-1, b].

### Copilot Tasks:

- Read this file and gather context from stop_position_ip.py,  vertical_diff_model.ipynb, and the rest of the repository where necessary.
- Create stop_position_dp.py and translate the IP solver to the DP program defined above.
- You will have to come up with the r_p^{LB} and r_p^{UB}. Be efficient with your choices.
- Overall, ensure the code is readable and non-redundant.