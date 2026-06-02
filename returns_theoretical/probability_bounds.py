import numpy as np
from scipy.optimize import minimize

# ----------------------------
# Objective (negative because scipy minimizes)
# ----------------------------

def obj(x):
    f, w1, w2, w3 = x

    num = np.exp(-3*f) + w1*np.exp(-f)
    den = np.exp(-3*f) + w1 + w2*np.exp(-f) + w3*np.exp(-2*f)

    return -(num/den)   # maximize -> minimize negative

# ----------------------------
# Constraints
# scipy uses g(x) >= 0
# ----------------------------

cons = [

    # w1 >= 1-exp(-f)
    {
        'type': 'ineq',
        'fun': lambda x: x[1] - (1 - np.exp(-x[0]))
    },

    # w2 >= 1-exp(-f)
    {
        'type': 'ineq',
        'fun': lambda x: x[2] - (1 - np.exp(-x[0]))
    },
    
    # w3 >= 1-exp(-f)
    {
        'type': 'ineq',
        'fun': lambda x: x[3] - (1 - np.exp(-x[0]))
    },

    # w2 <= w1(e^f-1)+(1-exp(-f))
    {
        'type': 'ineq',
        'fun': lambda x:
            x[1]*(np.exp(x[0])-1)
            + (1 - np.exp(-x[0]))
            - x[2]
    },
    
    # w2 <= w1(e^f-1)+(1-exp(-f))
    {
        'type': 'ineq',
        'fun': lambda x:
            (x[1] + x[2])*(np.exp(x[0])-1)
            + (1 - np.exp(-x[0]))
            - x[3]
    }

]

# ----------------------------
# Bounds
# ----------------------------

bounds = [
    (0.01,1),  # f
    (0,1),  # w1
    (0,1),  # w2
    (0,1)   # w3
]

# ----------------------------
# Initial guess
# ----------------------------

x0 = [0.5, 0.8, 0.6, 0.1]

# ----------------------------
# Solve
# ----------------------------

res = minimize(
    obj,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=cons
)

# ----------------------------
# Output
# ----------------------------

f_star, w1_star, w2_star, w3_star = res.x
ratio_star = -res.fun

print("Optimal f:", f_star)
print("Optimal w1:", w1_star)
print("Optimal w2:", w2_star)
print("Optimal w3:", w3_star)
print("Max ratio:", ratio_star)