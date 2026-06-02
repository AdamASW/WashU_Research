import ao_experiments as aoexp
import numpy as np

n1=5
f1 = 1.0
prices1 = [20,50,55,60,65]
utilities1 = [1.0,0.9,0.8,0.7,0.6]
return_costs = [1,1,1,1,1]

test1 = aoexp.AOExperiment(n1, f1, prices1, return_costs, utilities1)
profit, assortment = test1.optimizer_solve()
brute_profit, brute_assortment, _, _ = test1.brute_force_solve()

print("Optimal assortment:", assortment)
print("Profit:", profit)
print("Brute force assortment:", brute_assortment)
print("Brute profit:", max(brute_profit))

## Optimization Problem for N=2:
