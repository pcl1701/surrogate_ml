# for varying sets of parameters

import sys
sys.path.append("helper")

import numpy as np
from true_loss_funcs import generate_neg_log_likelihood_16_age
from skopt.learning import RandomForestRegressor
from skopt import Optimizer
from skopt.space import Real
from scipy.optimize import minimize
from tqdm import tqdm
import time
import pickle

path = "seir_16_age_results/varying_parms/"

rng1 = np.random.default_rng(123)

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
rf_nm_query_incs = []
rf_nm_true_loss = [] # each element should be a number
rf_nm_best_loss = [] # each element should have length len(query_incs)

print("RUNNING RANDOM FOREST + NELDER-MEAD")

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
    
        # initialize regressor and skopt optimizer
        search_space = [Real(0.0, 2.0) for _ in range(n_phis)]
        rf = RandomForestRegressor()
        opt = Optimizer(
            dimensions=search_space,
            base_estimator=rf,
            n_initial_points=1,
            initial_point_generator="lhs",
            acq_func="gp_hedge",
            acq_optimizer="sampling",
            acq_optimizer_kwargs={"n_points":10000}
        )

        n_iters = 500
        for k in range(n_iters):
            # perform one step of Bayesian optimization
            next_x = opt.ask()
            f_val = counted_loss_fn(next_x)
            opt.tell(next_x, f_val)

            # check if found better answer
            new_best_loss = np.min(opt.yi)
            if new_best_loss < cur_best_loss[0]:
                cur_best_loss[0] = new_best_loss
            
            # check if need to save answer
            if counted_loss_fn.n_calls >= query_incs[incs_counter[0]]:
                best_loss.append(cur_best_loss[0])
                incs_counter[0] += 1
            
            # check if done
            if incs_counter[0] == len(query_incs):
                reached_end[0] = True
                break

        # exit if done
        if reached_end[0]:
            break

        # get best loss point
        min_idx = np.argmin(opt.yi)
        x0 = opt.Xi[min_idx]

        def callback_fn(intermediate_result):
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

        # run Nelder-Mead
        result = minimize(fun=counted_loss_fn,
                          x0=x0,
                          method="Nelder-Mead",
                          callback=callback_fn,
                          bounds=[(0,2)]*16)

        # exit if done
        if reached_end[0]:
            break

    # save results
    rf_nm_query_incs.append(query_incs)
    rf_nm_true_loss.append(loss_fn(true_phis))
    rf_nm_best_loss.append(best_loss)

    # print("query_incs:", rf_nm_query_incs[-1])
    # print("best_loss:", rf_nm_best_loss[-1])
    # print("true_loss:", rf_nm_true_loss[-1])
    # print()

    # save results
    with open(path + "seir_16_age_rf_nm_query_incs.pkl", "wb") as f:
        pickle.dump(rf_nm_query_incs, f)
    with open(path + "seir_16_age_rf_nm_true_loss.pkl", "wb") as f:
        pickle.dump(rf_nm_true_loss, f)
    with open(path + "seir_16_age_rf_nm_best_loss.pkl", "wb") as f:
        pickle.dump(rf_nm_best_loss, f)

print("DONE")
