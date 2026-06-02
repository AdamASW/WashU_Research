import ao_experiments as aoexp
import numpy as np

n1 = 5
f1 = 0.2
prices1 = [13, 14, 16, 20, 18]
return_costs1 = [2, 3, 2, 3, 1]
utilities1 = [7, 6, 5, 4, 3]

n2 = 5
f2 = 0
prices2 = [100, 0, 0, 0, 0]
return_costs2 = [10, 10, 10, 10, 10]
utilities2 = [100, 1, 1, 1, 1]

## TEST 3 ##
# 1. Increasing prices with decreasing utilities
# 2. Lower f and low return costs.
# We get an optimality gap of ~2.4
############
n3 = 5
f3 = 1.0
utilities3 = [1.0, 0.9, 0.8, 0.7, 0.6]    
prices3 = [20, 50, 55, 60, 65]              
return_costs3 = [1, 1, 1, 1, 1]

test3 = aoexp.AOExperiment(n3, f3, prices3, return_costs3, utilities3)
test3_bulk_objectives, test3_best_bulk_S, test3_seq_objectives, test3_best_seq_S = test3.brute_force_solve()
print("Test 3 Max Profit Gap:" + str(max(test3_seq_objectives) - max(test3_bulk_objectives)))
print(test3_best_seq_S)
print(test3_best_bulk_S)
############

## TEST 4 ##
# LARGER VERSION OF TEST3
############
n4 = 10
f4 = 1.0
utilities4 = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]    
prices4 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]              
return_costs4 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 

test4 = aoexp.AOExperiment(n4, f4, prices4, return_costs4, utilities4)
test4_bulk_objectives, test4_best_bulk_S, test4_seq_objectives, test4_best_seq_S = test4.brute_force_solve()
print("Test 3 Max Profit Gap:" + str(max(test4_seq_objectives) - max(test4_bulk_objectives)))
print(test4_best_seq_S)
print(test4_best_bulk_S)
############

# test1 = aoexp.AOExperiment(n1, f1, prices1, return_costs1, utilities1)
# test1_bulk_objectives, _, test1_seq_objectives, _ = test1.brute_force_solve()

# test2 = aoexp.AOExperiment(n2, f2, prices2, return_costs2, utilities2)
# test2_bulk_objectives, _, test2_seq_objectives, _ = test2.brute_force_solve()

# print("Test 1 Max Profit Gap:" + str(max(test1_seq_objectives) - max(test1_bulk_objectives)))

# print("Test 2 Max Profit Gap:" + str(max(test2_seq_objectives) - max(test2_bulk_objectives)))

## TEST 5: MANY RUNS TEST ##:

n = 5
return_costs = [1] * n
num_tests = 1000

max_gap = -np.inf
best_instance = None

for test_idx in range(num_tests):
    # Sample random f in [0,2]
    f = np.random.uniform(0, 2)
    
    # Sample utilities (weights) in [0,2]
    utilities = np.random.uniform(0, 2, size=n)
    
    # Sample prices in [10,100]
    prices = np.random.uniform(10, 100, size=n)
    
    # Initialize experiment
    test = aoexp.AOExperiment(n, f, prices, return_costs, utilities)
    bulk_obj, bulk_best, seq_obj, seq_best = test.brute_force_solve()
    
    gap = max(seq_obj) - max(bulk_obj)
    
    if gap > max_gap:
        max_gap = gap
        best_instance = {
            "f": f,
            "utilities": utilities.copy(),
            "prices": prices.copy(),
            "bulk_best": bulk_best,
            "seq_best": seq_best,
            "gap": gap
        }

# Print the instance that gave the largest gap
print("Largest Optimality Gap:", best_instance["gap"])
print("f:", best_instance["f"])
print("Utilities:", best_instance["utilities"])
print("Prices:", best_instance["prices"])
print("Bulk Optimal Set:", best_instance["bulk_best"])
print("Sequential Optimal Set:", best_instance["seq_best"])



## TEST 6: ARBITRARY SCALING ##
n6 = 5
utilities6 = [2.0, 1.8, 1.5, 1.2, 1.0]
return_costs6 = [1, 1, 1, 1, 1]
f6 = 1.0

print("P\tBulk Profit\tSeq Profit\tGap")
for P in range(100, 10001, 100):
    prices6 = [10, 20, 30, P, P]
    
    test6 = aoexp.AOExperiment(n6, f6, prices6, return_costs6, utilities6)
    bulk_obj, bulk_best, seq_obj, seq_best = test6.brute_force_solve()
    
    bulk_profit = max(bulk_obj)
    seq_profit  = max(seq_obj)
    gap = ((seq_profit - bulk_profit )/ bulk_profit )*100
    
    print(f"{P}\t{bulk_profit:.2f}\t\t{seq_profit:.2f}\t\t{gap:.2f}")