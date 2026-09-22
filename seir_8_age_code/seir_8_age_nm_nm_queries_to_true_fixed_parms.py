# for a fixed set of parameters

import sys
sys.path.append("helper")

import numpy as np
from true_loss_funcs import generate_neg_log_likelihood_8_age
from tqdm import tqdm
from scipy.optimize import minimize
import pickle

path = "seir_8_age_results/fixed_parms/"

# create true loss function
rng = np.random.default_rng(123)
true_phis = [0.95737744, 0.67550791, 0.86565616, 0.94441362,
             0.74916310, 0.69556869, 0.57126105, 0.63265584]
loss_fn = generate_neg_log_likelihood_8_age(true_phis, rng)
true_loss = loss_fn(true_phis)

# class for keeping track of number of loss function calls
class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.n_calls = 0
    
    def __call__(self, x):
        self.n_calls += 1
        return self.func(x)

# arrays to store results
n_runs = 100

nm_nm_n_calls = []

print("RUNNING RANDOM FOREST + NELDER-MEAD")

for i in tqdm(range(n_runs)):

    n_calls = []
    reached_theshold = [False]

    while True:

        # reset the number of counts
        counted_loss_fn = FunctionCounter(loss_fn)

        def callback_fn1(intermediate_result):
            # check if reached threshold
            if intermediate_result.fun < true_loss:
                reached_theshold[0] = True
                raise StopIteration

        # start with 500 queries
        result1 = minimize(fun=counted_loss_fn,
                           x0=rng.uniform(low=0.0, high=2.0, size=8),
                           method="Nelder-Mead",
                           callback=callback_fn1,
                           bounds=[(0,2)]*8,
                           options={"maxfev":500})
        
        # exit if reached threshold
        if reached_theshold[0]:
            n_calls.append(counted_loss_fn.n_calls)
            break

        def callback_fn2(intermediate_result):
            # check if reached threshold
            if intermediate_result.fun < true_loss:
                reached_theshold[0] = True
                raise StopIteration

        # restart Nelder-Mead
        result2 = minimize(fun=counted_loss_fn,
                           x0=result1.x,
                           method="Nelder-Mead",
                           callback=callback_fn2,
                           bounds=[(0,2)]*8)
        
        # exit if reached threshold
        if reached_theshold[0]:
            n_calls.append(counted_loss_fn.n_calls)
            break

        n_calls.append(counted_loss_fn.n_calls)
    
    print("n_queries:", n_calls)
    print()

    # save all values
    nm_nm_n_calls.append(n_calls)

    # save results
    with open(path + "seir_8_age_nm_nm_queries_to_true_n_calls.pkl", "wb") as f:
        pickle.dump(nm_nm_n_calls, f)

print("DONE")
