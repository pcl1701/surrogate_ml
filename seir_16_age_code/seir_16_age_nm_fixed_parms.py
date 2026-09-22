# for a fixed set of parameters

import sys
sys.path.append("helper")

import numpy as np
from true_loss_funcs import generate_neg_log_likelihood_16_age
from tqdm import tqdm
from scipy.optimize import minimize
import pickle

path = "seir_16_age_results/fixed_parms/"

# create true loss function
rng = np.random.default_rng(123)
true_phis = [1.72386259, 1.15532378, 1.28441101, 1.4971796,
             0.95784338, 0.90884865, 1.59289604, 0.5372940,
             0.75863593, 1.59368178, 1.14030436, 1.30624542,
             0.88678266, 0.9912558 , 1.0009599 , 1.90459005]
loss_fn = generate_neg_log_likelihood_16_age(true_phis, rng)

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
                      x0=rng.uniform(low=0.0, high=2.0, size=16),
                      method="Nelder-Mead",
                      callback=callback_fn,
                      bounds=[(0,2)]*16,
                      options={"maxfev":1000, "xatol": -1, "fatol": -1})

    # save all values
    nm_n_calls.append(n_calls)
    nm_saved_x.append(saved_x)
    nm_min_nll.append(min_nll)

    # save results
    with open(path + "seir_16_age_nm_n_calls.pkl", "wb") as f:
        pickle.dump(nm_n_calls, f)
    with open(path + "seir_16_age_nm_saved_x.pkl", "wb") as f:
        pickle.dump(nm_saved_x, f)
    with open(path + "seir_16_age_nm_min_nll.pkl", "wb") as f:
        pickle.dump(nm_min_nll, f)

print("DONE")
