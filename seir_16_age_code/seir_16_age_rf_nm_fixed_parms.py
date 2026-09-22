# for a fixed set of parameters

import sys
sys.path.append("helper")

import numpy as np
from skopt.learning import RandomForestRegressor
from skopt import Optimizer
from skopt.space import Real
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

rf_nm_n_calls = []
rf_nm_saved_x = []
rf_nm_min_nll = []

print("RUNNING RANDOM FOREST + NELDER-MEAD")

for i in tqdm(range(n_runs)):

    counted_loss_fn = FunctionCounter(loss_fn)
    n_calls = []
    saved_x = []
    min_nll = []

    n_phis = 16
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

    n_queries = 500
    for k in range(n_queries):
        # perform one step of Bayesian optimization
        next_x = opt.ask()
        f_val = counted_loss_fn(next_x)
        opt.tell(next_x, f_val)
        n_calls.append(counted_loss_fn.n_calls)
        min_idx = np.argmin(opt.yi)
        saved_x.append(opt.Xi[min_idx])
        min_nll.append(opt.yi[min_idx])

    # get best loss point
    min_idx = np.argmin(opt.yi)
    x0 = opt.Xi[min_idx]

    def callback_fn(intermediate_result):
        n_calls.append(counted_loss_fn.n_calls)
        saved_x.append(intermediate_result.x)
        min_nll.append(intermediate_result.fun)

    result = minimize(fun=counted_loss_fn,
                      x0=x0,
                      method="Nelder-Mead",
                      callback=callback_fn,
                      bounds=[(0,2)]*16,
                      options={"maxfev":500, "xatol": -1, "fatol": -1})

    # save all values
    rf_nm_n_calls.append(n_calls)
    rf_nm_saved_x.append(saved_x)
    rf_nm_min_nll.append(min_nll)

    # save results
    with open(path + "seir_16_age_rf_nm_n_calls.pkl", "wb") as f:
        pickle.dump(rf_nm_n_calls, f)
    with open(path + "seir_16_age_rf_nm_saved_x.pkl", "wb") as f:
        pickle.dump(rf_nm_saved_x, f)
    with open(path + "seir_16_age_rf_nm_min_nll.pkl", "wb") as f:
        pickle.dump(rf_nm_min_nll, f)

print("DONE")
