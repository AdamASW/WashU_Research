import numpy as np
from scipy.optimize import differential_evolution
import ao_experiments as aoexp

############################
# Problem size
############################
n = 2

############################
# Objective function
############################
def worst_case_objective(theta):

    # unpack parameters
    f = theta[0]
    prices = theta[1:1+n]
    utilities = theta[1+n:1+2*n]
    
    # --- enforce descending utilities ---
    order = np.argsort(-utilities)
    utilities = utilities[order]
    prices = prices[order]

    return_costs = [1]*n

    try:

        exp = aoexp.AOExperiment(n, f, prices, return_costs, utilities)

        bulk_obj, _, seq_obj, _, _, _, _, _ = exp.brute_force_solve()
        # bulk_obj, _ = exp.optimizer_solve() # <== Optimizer

        #bulk_profit = bulk_obj # <== Optimizer
        bulk_profit = max(bulk_obj)
        seq_profit = max(seq_obj)

        # avoid divide by zero
        if bulk_profit <= 1e-6:
            return 1e6

        # adding a floor to prevent blow-ups:
        ratio = seq_profit / max(bulk_profit, 1e-3)

        # negative because scipy minimizes
        return -ratio

    except:
        return 1e6


############################
# Parameter bounds
############################

bounds = []

# f parameter
bounds.append((0.0, 1.0))

# prices
for _ in range(n):
    bounds.append((1, 30))

# utilities
for _ in range(n):
    bounds.append((0, 1))


############################
# Run optimizer
############################

result = differential_evolution(
    worst_case_objective,
    bounds,
    maxiter=2000,
    popsize=20,
    seed=np.random.randint(1,1000001),
    polish=True
)

############################
# Extract best instance
############################

theta = result.x

f_best = theta[0]
prices_best = theta[1:1+n]
utilities_best = theta[1+n:1+2*n]

return_costs = [1]*n

exp = aoexp.AOExperiment(n, f_best, prices_best, return_costs, utilities_best)

bulk_obj, bulk_best, seq_obj, seq_best, bulk_probs, seq_probs, seq_nokeep, bulk_nokeep = exp.brute_force_solve()

bulk_profit = max(bulk_obj)
seq_profit = max(seq_obj)
gap_ratio = seq_profit / bulk_profit

############################
# Print results
############################

print("\nWorst Instance Found")
print("-----------------------")

print("f:", f_best)
print("prices:", prices_best)
print("utilities:", utilities_best)

print("\nBulk best assortment:", bulk_best)
print("Sequential best assortment:", seq_best)

print("\nBulk profit:", bulk_profit)
print("Sequential profit:", seq_profit)

print("\nWorst-case ratio:", gap_ratio)

print("\nChoice Probability ratios:", bulk_probs)
print(" | Bulk No-Purchase Prob:", bulk_nokeep)
print("\nChoice Probability ratios:", seq_probs)
print(" | Seq No-Purchase Prob:", seq_nokeep)