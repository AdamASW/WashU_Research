# AI IP Notes

## Core Objective
- The IP is used to recover the consideration set boundary (stopping position) from MNL-implied item weights.
- A predicted stop at position p implies predicted consideration set {1, ..., p}.
- Current fitting objective: maximize correctly predicted stopping positions across sessions.

## From MNL to IP Inputs
- Estimate MNL coefficients beta_hat from Expedia features.
- Compute utility per session-position: V_qp = x_qp' beta_hat (with the same feature scaling used in training).
- Convert utility to MNL weight: w_qp = exp(V_qp).
- Use w_qp as fixed inputs to the IP.

## IP Structure (First Prototype)
- Decision variables:
  - r_p: reservation threshold by position (monotone decreasing in p).
  - t_qp in {0,1}: trigger for whether w_qp clears threshold r_p.
  - z_qp in {0,1}: predicted stopping position (first triggered position).
- Constraints enforce:
  - monotone thresholds,
  - trigger logic,
  - first-trigger stopping rule,
  - exactly one predicted stop per session.
- Objective maximizes stop-position recovery accuracy.

## Why Learned Reservation Thresholds Matter
- Behavioral interpretation: position-specific strictness of acceptance.
- Generalization: one compact threshold profile for many sessions.
- Counterfactual use: test how ranking/feature shifts move stopping and consideration sets.
- Integration: creates a consideration layer before purchase-choice modeling.

## 0-Likelihood Issue (Professor's Point)
- If purchased item i is outside predicted consideration set C_q, then P(i | C_q) = 0.
- This causes log-likelihood contribution = -infinity for that case.

## Agreed Handling Policy
- Keep a count of 0-likelihood cases as a reported diagnostic.
- Exclude (filter out) 0-likelihood sessions from optimization fitting so they are not used to estimate reservation thresholds.
- In evaluation, report both:
  - count/rate of 0-likelihood cases,
  - metrics on the retained (non-zero-likelihood) subset.

## Two Practical Implementation Options (for later)
- Option 1: One-pass filter after initial fit.
  - Fit thresholds on all sessions once.
  - Identify/count 0-likelihood sessions under that fit.
  - Filter those sessions out for reported evaluation on the retained subset.
- Option 2: Iterative filtered re-fit.
  - Fit thresholds on all sessions.
  - Identify/count and remove 0-likelihood sessions.
  - Re-fit thresholds on the retained subset.
  - Optionally repeat until the retained set stabilizes.

## IP Formulation (Main Version)
Recall that for each session s and position i, w_si = h_si \dot \beta_hat, where h_si is the hotel listing in position i of search s and \beta_hat is the output from the MNL estimation.

We now build an IP that maximizes the amount of search stopping points it can correctly predict by deciding on a vector of reservation 'prices' for each of the k customer types that best explain the data. As seen in the following formulation, the inclusion of k customer types as a parameter allows for the opportunity for more customer behaviors to be captured in the modeling.

Let W_sp = (\sum_{i=1}^p w_si).
Then, the stopping point for a search s is T_s min{p : W_sp >= r_pk}. To clarify, T_s is observed in the expedia data.

The formulation is then as follows:

max \sum_{k \in K} (\sum_{s \in S} x_sk)

s.t.

(1) [Monotonicty Constraint] r_pk >= r_{p+1}k \forall k \in K, p = 1,...,N-1
(2) [Pre-stopping condition] W_sp <= r_pk - \epsilon + M(1 - x_sk) \forall s \in S, k \in K, p < T_s
(3) [Stopping condition] W_{s{T_s}} >= r_{{T_s}k} - M(1 - x_sk) \forall s \in S, k \in K
(4) [Only 1 k can solve] \sum_{k \in K} x_sk <= 1 \forall s \in S
(5) [Reservation scale] min_s{W_sp} <= r_pk <= max_s{W_sp} \forall k \in K, p = 1,...,N
(6) [Binary constraint] x_sk \in {0,1} \forall s \in S, k \in K
(7) [Symmetry-breaking] r_1k >= r_1{k+1} \forall k = 1,...,K-1

Computing M:
For constraint (2), I reccomend computing M as M_{spk}^{pre} = W_sp - min_s{W_sp} + \epsilon.
For constraint (3), I reccomend computing M as M_{sk}^{stop} = max_s{r_{{T_s}k}} - W_{s{T_s}}.
These values will tighten the Big-M values and ease the IP solves.

Considerations:
- Ensure weights are computed per hotel per search using vertical diff MNL estimated beta output.
- Filter out searches in the data which don't have a click_bool = 1 (we need an observed T_s value).
- Choose a value of \epsilon that best fits the output of the MNL estimation.