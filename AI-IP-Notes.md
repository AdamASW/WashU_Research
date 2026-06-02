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
