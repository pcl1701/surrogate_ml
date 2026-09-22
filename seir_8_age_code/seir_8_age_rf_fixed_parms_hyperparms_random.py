# for a fixed set of parameters

import sys
sys.path.append("helper")

import numpy as np
from skopt.learning import RandomForestRegressor
from skopt import Optimizer
from skopt.space import Real
from true_loss_funcs import generate_neg_log_likelihood_8_age
from tqdm import tqdm
import time
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
n_runs = 10000

rf_n_calls = []
rf_saved_x = []
rf_min_nll = []
rf_n_trees = []
rf_times = []

print("RUNNING RANDOM FOREST WITH VARYING NUMBER OF TREES")
rng = np.random.default_rng(123)

for i in range(n_runs):

    counted_loss_fn = FunctionCounter(loss_fn)
    n_calls = []
    saved_x = []
    min_nll = []

    n_phis = 8
    search_space = [Real(0.0, 2.0) for _ in range(n_phis)]

    # choose number of trees (between 1 and 1000), spaced uniformly on log scale
    x = 10 ** rng.uniform(0, 3)   # since log10(1)=0, log10(1000)=3
    n = int(np.round(x))
    # ensure the result is within [1, 1000]
    n_trees = max(1, min(1000, n))

    print("N_TREES:", n_trees)

    t_start = time.time()

    # initialize regressor and skopt optimizer
    rf = RandomForestRegressor(n_estimators=n_trees)
    opt = Optimizer(
        dimensions=search_space,
        base_estimator=rf,
        n_initial_points=1,
        initial_point_generator="lhs",
        acq_func="gp_hedge",
        acq_optimizer="sampling",
        acq_optimizer_kwargs={"n_points":10000}
    )

    n_queries = 1000
    for j in range(n_queries):
        # perform one step of Bayesian optimization
        next_x = opt.ask()
        f_val = counted_loss_fn(next_x)
        opt.tell(next_x, f_val)
        n_calls.append(counted_loss_fn.n_calls)
        min_idx = np.argmin(opt.yi)
        saved_x.append(opt.Xi[min_idx])
        min_nll.append(opt.yi[min_idx])

    t_end = time.time()
    t_total = t_end - t_start
    print("t_total", t_total)

    # save all values
    rf_n_calls.append(n_calls)
    rf_saved_x.append(saved_x)
    rf_min_nll.append(min_nll)
    rf_n_trees.append(n_trees)
    rf_times.append(t_total)

    # save results
    with open(path + "seir_8_age_rf_n_calls_hyperparms_random.pkl", "wb") as f:
        pickle.dump(rf_n_calls, f)
    with open(path + "seir_8_age_rf_saved_x_hyperparms_random.pkl", "wb") as f:
        pickle.dump(rf_saved_x, f)
    with open(path + "seir_8_age_rf_min_nll_hyperparms_random.pkl", "wb") as f:
        pickle.dump(rf_min_nll, f)
    with open(path + "seir_8_age_rf_n_trees_hyperparms_random.pkl", "wb") as f:
        pickle.dump(rf_n_trees, f)
    with open(path + "seir_8_age_rf_times_hyperparms_random.pkl", "wb") as f:
        pickle.dump(rf_times, f)

print("DONE")
