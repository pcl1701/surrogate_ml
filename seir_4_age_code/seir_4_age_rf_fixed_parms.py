# for a fixed set of parameters

import sys
sys.path.append("helper")

import numpy as np
from skopt.learning import RandomForestRegressor
from skopt import Optimizer
from skopt.space import Real
from true_loss_funcs import generate_neg_log_likelihood_4_age
from tqdm import tqdm
import pickle

path = "seir_4_age_results/fixed_parms/"

# create true loss function
rng = np.random.default_rng(123)
true_phis = [0.97959427, 0.69127489, 0.6611171, 0.71035546]
loss_fn = generate_neg_log_likelihood_4_age(true_phis, rng)

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

rf_n_calls = []
rf_saved_x = []
rf_min_nll = []

print("RUNNING RANDOM FOREST")

for i in tqdm(range(n_runs)):

    counted_loss_fn = FunctionCounter(loss_fn)
    n_calls = []
    saved_x = []
    min_nll = []

    n_phis = 4
    search_space = [Real(0.0, 2.0) for _ in range(n_phis)]

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

    # save all values
    rf_n_calls.append(n_calls)
    rf_saved_x.append(saved_x)
    rf_min_nll.append(min_nll)

    # save results
    with open(path + "seir_4_age_rf_n_calls.pkl", "wb") as f:
        pickle.dump(rf_n_calls, f)
    with open(path + "seir_4_age_rf_saved_x.pkl", "wb") as f:
        pickle.dump(rf_saved_x, f)
    with open(path + "seir_4_age_rf_min_nll.pkl", "wb") as f:
        pickle.dump(rf_min_nll, f)

print("DONE")
