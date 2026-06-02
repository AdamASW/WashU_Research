# Context

# Relevant Research
- returns_emprical/vertical.pdf is the most important paper. It describes the veritcally differentiated MNL model. Read the model and estimation sections.
- The paper titled "TEX" is about the click-based MNL.  See Appendix E.C.7.1 for how to write down the log likelihood under the MNL model.
- The paper titled platform ranking is the one I want to estimate. Just read the model section.

# Current Goal/Task:
- So far, in 'vertical_diff_model.ipynb', we implemented a vertical differentiated MNL model and estimated the parameters using MLE.
- Next, we will write an Integer Program formulation whose decision variables are reservation "weight" decreasing in the position. We will also have a cumulative "weight" variable that is the sum of the reservation weights up to that position.
- We will have a decision variable Z_p (0,1) that is triggered if the decided weight is higher (than unsure) and p s.t. Z_p = 1 is the stopping point, and we will use this to check if p was the actual stopping point, and the goal of the optimization is to maximize the correct number of stopping points.
- Eventually, we will modify the formulation to include multiple types of applicable reservation prices or customer types.
- Then, we will compare OOS LL, Hit Rate, and Avg Rank of the IP results.

# Current Questions/Challenges:
- Cumulative weight definition (recommended):
	- Let q index search sessions and p index ranked positions.
	- Compute product weights from estimated utility as w_{q,p} = exp(V_{q,p}), where V_{q,p} = x_{q,p}' beta_hat.
	- Let r_p be a position-specific reservation threshold (decreasing with position: r_{p+1} <= r_p).
	- If cumulative thresholds are required, define R_p = sum_{k=1..p} r_k and compare w_{q,p} against R_p.
	- If the model intends a direct threshold by position, then R_p is not needed; use only r_p with monotonicity.
	- In the AO code, the natural cumulative term is over prior product weights, not prior thresholds: S_{q,p} = sum_{k < p} w_{q,k} * x_{q,k}. This produces threshold T_{q,p} = (exp(f)-1) * S_{q,p} + (1-exp(-f)).
- Weight construction from MNL model:
	- After MLE in vertical_diff_model.ipynb, beta_hat maps features to utility: V_{q,p} = x_{q,p}' beta_hat.
	- Convert utility to MNL weight via w_{q,p} = exp(V_{q,p}).
	- Outside option / no-purchase weight is w0 = exp(-f) (same structure used in returns_theoretical code).
	- Choice probabilities then follow P(i | C_q) = w_{q,i} / (w0 + sum_{j in C_q} w_{q,j}).
	- For the stopping-IP target, create labels y_{q,p}=1 if p is the observed stopping position (deepest click or booked item), else 0.
	- Use binary z_{q,p} to indicate predicted stop and maximize sum_{q,p} y_{q,p} * z_{q,p}.
- Suggested immediate implementation:
	- Start with direct position thresholds r_p and constraints r_{p+1} <= r_p.
	- Add trigger binaries t_{q,p}=1{w_{q,p} >= r_p} using Big-M linearization.
	- Define stop binaries z_{q,p} as first triggered position and enforce one stop per session.
	- Then test a second formulation where threshold uses AO-style cumulative product-weight term S_{q,p}.