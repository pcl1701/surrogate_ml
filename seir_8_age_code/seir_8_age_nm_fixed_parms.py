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

nm_n_calls = []
nm_saved_x = []
nm_min_nll = []

print("RUNNING NELDER-MEAD")

for i in tqdm(range(n_runs)):

    counted_loss_fn = FunctionCounter(loss_fn)
    n_calls = []
    saved_x = []
    min_nll = []

    def callback_fn(intermediate_result):
        n_calls.append(counted_loss_fn.n_calls)
        saved_x.append(intermediate_result.x)
        min_nll.append(intermediate_result.fun)

    result = minimize(fun=counted_loss_fn,
                      x0=rng.uniform(low=0.0, high=2.0, size=8),
                      method="Nelder-Mead",
                      callback=callback_fn,
                      bounds=[(0,2)]*8,
                      options={"maxfev":1000, "xatol": -1, "fatol": -1})

    # save all values
    nm_n_calls.append(n_calls)
    nm_saved_x.append(saved_x)
    nm_min_nll.append(min_nll)

    # save results
    with open(path + "seir_8_age_nm_n_calls.pkl", "wb") as f:
        pickle.dump(nm_n_calls, f)
    with open(path + "seir_8_age_nm_saved_x.pkl", "wb") as f:
        pickle.dump(nm_saved_x, f)
    with open(path + "seir_8_age_nm_min_nll.pkl", "wb") as f:
        pickle.dump(nm_min_nll, f)

print("DONE")
