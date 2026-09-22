import sys
sys.path.append("helper")

import numpy as np
from true_loss_funcs import generate_neg_log_likelihood_pitzer_6
from tqdm import tqdm
from scipy.optimize import minimize
import time
import pickle

path = "pitzer_results/"

# create true loss function
rng = np.random.default_rng(123)
loss_fn = generate_neg_log_likelihood_pitzer_6(rng)

# class for keeping track of number of loss function calls
class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.n_calls = 0
    
    def __call__(self, x):
        self.n_calls += 1
        return self.func(x)

# arrays to store results
n_runs = 10_000

nm_nm_n_calls = []
nm_nm_times = []
nm_nm_best_x = []
nm_nm_best_loss = []
n_parms = 6

print("NELDER-MEAD + NELDER-MEAD")

for i in tqdm(range(n_runs)):

    # prepare to store results
    counted_loss_fn = FunctionCounter(loss_fn)
    n_calls = []
    times = []
    best_x = []
    best_loss = []

    # keep track of time
    start_time = [time.time()]

    # keep track of best values
    curr_best_loss = [float("inf")]
    curr_best_x = [0]

    def callback_fn1(intermediate_result):
        elapsed_time = time.time() - start_time[0]

        # check if found better answer
        new_best_loss = intermediate_result.fun
        if new_best_loss < curr_best_loss[0]:
            curr_best_loss[0] = new_best_loss
            curr_best_x[0] = intermediate_result.x
        
        # save results
        n_calls.append(counted_loss_fn.n_calls)
        times.append(elapsed_time)
        best_x.append(curr_best_x[0])
        best_loss.append(curr_best_loss[0])

    # first run 500 iterations of Nelder-Mead
    result1 = minimize(fun=counted_loss_fn,
                       x0=rng.uniform(low=0.0, high=1.0, size=n_parms),
                       method="Nelder-Mead",
                       callback=callback_fn1,
                       bounds=[(0,1)]*n_parms,
                       options={"maxfev":500})

    # get best loss point
    x0 = result1.x

    def callback_fn2(intermediate_result):
        elapsed_time = time.time() - start_time[0]

        # check if found better answer
        new_best_loss = intermediate_result.fun
        if new_best_loss < curr_best_loss[0]:
            curr_best_loss[0] = new_best_loss
            curr_best_x[0] = intermediate_result.x
        
        # save results
        n_calls.append(counted_loss_fn.n_calls)
        times.append(elapsed_time)
        best_x.append(curr_best_x[0])
        best_loss.append(curr_best_loss[0])

    result2 = minimize(fun=counted_loss_fn,
                       x0=x0,
                       method="Nelder-Mead",
                       callback=callback_fn2,
                       bounds=[(0,1)]*n_parms)

    nm_nm_n_calls.append(n_calls)
    nm_nm_times.append(times)
    nm_nm_best_x.append(best_x)
    nm_nm_best_loss.append(best_loss)

    # save results
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_nm_nm_n_calls.pkl", "wb") as f:
        pickle.dump(nm_nm_n_calls, f)
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_nm_nm_times.pkl", "wb") as f:
        pickle.dump(nm_nm_times, f)
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_nm_nm_best_x.pkl", "wb") as f:
        pickle.dump(nm_nm_best_x, f)
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_nm_nm_best_loss.pkl", "wb") as f:
        pickle.dump(nm_nm_best_loss, f)

print("DONE")