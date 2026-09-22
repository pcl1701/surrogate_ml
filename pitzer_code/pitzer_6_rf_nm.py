import sys
sys.path.append("helper")

import numpy as np
from skopt.learning import RandomForestRegressor
from skopt import Optimizer
from skopt.space import Real
from scipy.optimize import minimize
from true_loss_funcs import generate_neg_log_likelihood_pitzer_6
from tqdm import tqdm
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

n_parms = 6
search_space = [Real(0.0, 1.0) for _ in range(n_parms)]

# arrays to store results
n_runs = 10_000

rf_nm_n_calls = []
rf_nm_times = []
rf_nm_best_x = []
rf_nm_best_loss = []

print("RUNNING RANDOM FOREST + NELDER-MEAD")

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

    # initialize regressor and skopt optimizer
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

    n_queries = 500
    for k in range(n_queries):
        # perform one step of Bayesian optimization
        next_x = opt.ask()
        f_val = counted_loss_fn(next_x)
        opt.tell(next_x, f_val)

        elapsed_time = time.time() - start_time[0]

        # check if found better answer
        new_best_loss = f_val
        if new_best_loss < curr_best_loss[0]:
            curr_best_loss[0] = new_best_loss
            curr_best_x[0] = next_x
        
        # save results
        n_calls.append(counted_loss_fn.n_calls)
        times.append(elapsed_time)
        best_x.append(curr_best_x[0])
        best_loss.append(curr_best_loss[0])

    # get best loss point
    min_idx = np.argmin(opt.yi)
    x0 = opt.Xi[min_idx]

    def callback_fn(intermediate_result):
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

    result = minimize(fun=counted_loss_fn,
                      x0=x0,
                      method="Nelder-Mead",
                      callback=callback_fn,
                      bounds=[(0,1)]*n_parms)
    
    rf_nm_n_calls.append(n_calls)
    rf_nm_times.append(times)
    rf_nm_best_x.append(best_x)
    rf_nm_best_loss.append(best_loss)

    # save results
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_rf_nm_n_calls.pkl", "wb") as f:
        pickle.dump(rf_nm_n_calls, f)
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_rf_nm_times.pkl", "wb") as f:
        pickle.dump(rf_nm_times, f)
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_rf_nm_best_x.pkl", "wb") as f:
        pickle.dump(rf_nm_best_x, f)
    with open(path + "pitzer_" + str(n_parms) + "_unbounded_rf_nm_best_loss.pkl", "wb") as f:
        pickle.dump(rf_nm_best_loss, f)

print("DONE")
