# for varying sets of parameters

import sys
sys.path.append("helper")

import numpy as np
from true_loss_funcs import generate_neg_log_likelihood_16_age
from scipy.optimize import minimize
from tqdm import tqdm
import time
import pickle

path = "seir_16_age_results/varying_parms/"

rng1 = np.random.default_rng(123)
rng2 = np.random.default_rng(456)

# class for keeping track of number of loss function calls
class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.n_calls = 0
    
    def __call__(self, x):
        self.n_calls += 1
        return self.func(x)

n_parms = 100
n_phis = 16

# arrays to store results
query_incs = [1000, 2000, 3000, 4000, 5000, 6000]
nm_query_incs = []
nm_true_loss = [] # each element should be a number
nm_best_loss = [] # each element should have length len(query_incs)

print("RUNNING NELDER-MEAD")

for i in tqdm(range(n_parms)):

    # generate true loss function
    true_phis = rng1.uniform(low=0.0, high=2.0, size=(n_phis,1))
    loss_fn = generate_neg_log_likelihood_16_age(true_phis, rng1)
    counted_loss_fn = FunctionCounter(loss_fn)

    # prepare array to store results
    best_loss = []

    incs_counter = [0]
    cur_best_loss = [1e+20]
    reached_end = [False]

    while True:

        def callback_fn1(intermediate_result):
            # check if found better answer
            if intermediate_result.fun < cur_best_loss[0]:
                cur_best_loss[0] = intermediate_result.fun
            
            # check if need to save answer
            if counted_loss_fn.n_calls >= query_incs[incs_counter[0]]:
                best_loss.append(cur_best_loss[0])
                incs_counter[0] += 1

            # check if done
            if incs_counter[0] == len(query_incs):
                reached_end[0] = True
                raise StopIteration

        # first run 500 queries with Nelder-Mead
        result1 = minimize(fun=counted_loss_fn,
                           x0=rng2.uniform(low=0.0, high=2.0, size=16),
                           method="Nelder-Mead",
                           callback=callback_fn1,
                           bounds=[(0,2)]*16,
                           options={"maxfev":500})
        
        # exit if done
        if reached_end[0]:
            break
        
        def callback_fn2(intermediate_result):
            # check if found better answer
            if intermediate_result.fun < cur_best_loss[0]:
                cur_best_loss[0] = intermediate_result.fun
            
            # check if need to save answer
            if counted_loss_fn.n_calls >= query_incs[incs_counter[0]]:
                best_loss.append(cur_best_loss[0])
                incs_counter[0] += 1

            # check if done
            if incs_counter[0] == len(query_incs):
                reached_end[0] = True
                raise StopIteration

        # run another Nelder-Mead
        result2 = minimize(fun=counted_loss_fn,
                           x0=result1.x,
                           method="Nelder-Mead",
                           callback=callback_fn2,
                           bounds=[(0,2)]*16)
        
        # exit if done
        if reached_end[0]:
            break

    # save results
    nm_query_incs.append(query_incs)
    nm_true_loss.append(loss_fn(true_phis))
    nm_best_loss.append(best_loss)

    # print("query_incs:", nm_query_incs[-1])
    # print("best_loss:", nm_best_loss[-1])
    # print("true_loss:", nm_true_loss[-1])
    # print()

    # save results
    with open(path + "seir_16_age_nm_nm_query_incs.pkl", "wb") as f:
        pickle.dump(nm_query_incs, f)
    with open(path + "seir_16_age_nm_nm_true_loss.pkl", "wb") as f:
        pickle.dump(nm_true_loss, f)
    with open(path + "seir_16_age_nm_nm_best_loss.pkl", "wb") as f:
        pickle.dump(nm_best_loss, f)

print("DONE")
