import numpy as np
from itertools import combinations
import math
import gurobipy as gp
from gurobipy import GRB

class AOExperiment:
    
    def __init__(self, n, f, prices, return_costs, utilities):
        """Initialize a Bulk AO Model solved via brute force. Ensure
        that utilities are sorted in order of decreasing utility.

        Args:
            n (int): assortment (S) length
            f (int): disutility parameter
            prices (list): prices for each product in S
            return_costs (list): return costs for each product
            utilities (_type_): _description_
        """
        self.n = n
        self.prices = np.array(prices)
        self.return_costs = np.array(return_costs)
        self.utilities = np.array(utilities)
        self.weights = np.exp(self.utilities)
        self.f = f
    
    def brute_force_solve(self):
        n = self.n
        best_bulk_profit = -np.inf
        bulk_best_S = None
        best_seq_profit = -np.inf
        seq_best_S = None
        bulk_objectives = []
        seq_objectives = []
        best_bulk_probs = []
        best_seq_probs = []

        # Compute l_Q (jmax)
        l_Q = 0
        for i in range(self.n):
            if self.weights[i] >= 1 - math.exp(-self.f):
                l_Q += 1
            else:
                break
        # print("LQ: " + str(l_Q))

        # Iterate over all subsets
        for r in range(n + 1):
            for S_indices in combinations(range(n), r):

                if len(S_indices) == 0:
                    bulk_profit = 0
                    seq_profit = 0

                else:

                    # --------------------
                    # BULK SECTION
                    # --------------------

                    ls = -1
                    for i, idx in enumerate(S_indices):
                        rhs_bulk = (
                            sum(self.weights[j] for j in S_indices[:i])
                            * (math.exp(self.f) - 1)
                            + (1 - math.exp(-self.f))
                        )
                        if self.weights[idx] < rhs_bulk:
                            break
                        ls = i
                    O_indices = S_indices[: ls+1]
                    # print("ls:" + str(ls))

                    if len(O_indices) == 0:
                        bulk_profit = 0
                        bulk_probs = {}
                        bulk_no_purchase = 0
                    else:
                        denom = math.exp(-self.f) + sum(self.weights[j] for j in O_indices)
                        bulk_probs = {
                            j: self.weights[j] / denom
                            for j in O_indices
                        }

                        bulk_profit = (
                            sum(
                                (self.prices[j] + self.return_costs[j])
                                * (self.weights[j] / denom)
                                for j in O_indices
                            )
                            - sum(self.return_costs[j] for j in O_indices)
                        )
                        
                        bulk_no_purchase = math.exp(-self.f) / denom

                    # --------------------
                    # SEQUENTIAL SECTION
                    # --------------------
                    
                    prefix = [j for j in S_indices if j <= l_Q]

                    if len(prefix) == 0:
                        seq_profit = 0
                        seq_probs = {}
                        prob_no_keep = 0
                    else:
                        # Compute denominator for π(i, prefix)
                        denom = math.exp(-self.f) + sum(self.weights[j] for j in prefix)

                        # Compute keep probabilities π(i, prefix)
                        seq_probs = {}
                        for i in prefix:
                            seq_probs[i] = self.weights[i] / denom

                        # Probability of keeping nothing
                        prob_no_keep = math.exp(-self.f) / denom

                        seq_profit = 0

                        # Case 1: keep product i
                        for i in prefix:
                            # Return cost of products tried BEFORE i
                            return_cost_before_i = sum(
                                self.return_costs[j] for j in prefix if j < i
                            )

                            seq_profit += seq_probs[i] * (
                                self.prices[i] - return_cost_before_i
                            )

                        # Case 2: keep nothing
                        total_return_cost = sum(self.return_costs[j] for j in prefix)

                        seq_profit -= prob_no_keep * total_return_cost

                    # --------------------
                    # BEST UPDATES
                    # --------------------

                    if bulk_profit > best_bulk_profit:
                        best_bulk_profit = bulk_profit
                        bulk_best_S = S_indices
                        best_bulk_probs = bulk_probs
                        best_bulk_nokeep = bulk_no_purchase

                    if seq_profit > best_seq_profit:
                        best_seq_profit = seq_profit
                        seq_best_S = S_indices
                        best_seq_probs = seq_probs
                        best_seq_nokeep = prob_no_keep
                        
                    bulk_objectives.append(bulk_profit)
                    seq_objectives.append(seq_profit)

        return bulk_objectives, bulk_best_S, seq_objectives, seq_best_S, best_bulk_probs, best_seq_probs, best_seq_nokeep, best_bulk_nokeep
    
    def optimizer_solve(self, f_trip=0):
        """Implementing the efficient enumeration / IP combination implemented
        on Page 16 of paper to solve AO-Bulk more efficiently.

        Args:
            f_trip (int, optional): Disutility of an entire trip. Defaults to 0.

        Returns:
            best_profit, best_set: returns the optimal assortment and corresponding
            maximized objective.
        """
        f_time = self.f
        n = len(self.prices)
        best_profit = -1e18
        best_set = None
        
        M = 10000
        
        for i_min in range(n):
            for k in range(1, n+1):
                model = gp.Model()
                model.setParam('OutputFlag', 0)
                
                x = model.addVars(n, vtype=GRB.BINARY, name="x")
                z = model.addVars(n, lb=0, name="z")
                y = model.addVar(lb=0, name="y")
                
                model.setObjective(
                    gp.quicksum((self.prices[i] + self.return_costs[i])*self.weights[i]*z[i] for i in range(n))
                    - self.return_costs[0]*k, # return costs all same by assumption.
                    GRB.MAXIMIZE
                )
                
                # cardinality constraint:
                model.addConstr(gp.quicksum(x[i] for i in range(n)) == k)
                
                # enforce i_min as smallest index chosen:
                model.addConstr(x[i_min] == 1)
                for j in range(i_min):
                    model.addConstr(x[j] == 0)
                
                # w(x)
                w_expr = gp.quicksum(self.weights[i]*x[i] for i in range(n))
                
                # constraint (1)
                lhs = w_expr*np.exp(-f_time*(k-1) - f_trip) + np.exp(-f_time*k - f_trip)
                rhs = self.weights[i_min] + np.exp(-(f_time + f_trip))
                model.addConstr(lhs >= rhs)
                
                # constraint (2)
                for i in range(n):
                    lhs = M*(1-x[i]) + self.weights[i]*x[i]
                    rhs = gp.quicksum(self.weights[j]*x[j]*(np.exp(f_time)-1) for j in range(i))
                    + (1-np.exp(-f_time))
                model.addConstr(lhs >= rhs)
                
                # constraint (4-7)
                model.addConstr(
                    y*np.exp(-f_time) + gp.quicksum(self.weights[i]*z[i] for i in range(n)) == 1
                )
                for i in range(n):
                    model.addConstr(y - z[i] <= M - M*x[i])
                    model.addConstr(z[i] <= M*x[i])
                    model.addConstr(z[i] <= y)
                
                model.optimize()
                
                if model.status == GRB.OPTIMAL:
                    profit = model.objVal
                    if profit > best_profit:
                        best_profit = profit
                        best_set = tuple(i for i in range(n) if x[i].X > 0.5)
        
        return best_profit, best_set