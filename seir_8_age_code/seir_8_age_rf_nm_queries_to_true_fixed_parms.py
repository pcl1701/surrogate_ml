# for a fixed set of parameters

import sys
sys.path.append("helper")

import numpy as np
from skopt.learning import RandomForestRegressor
from skopt import Optimizer
from skopt.space import Real
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

rf_nm_n_calls = []

print("RUNNING RANDOM FOREST + NELDER-MEAD")

for i in tqdm(range(n_runs)):

    n_calls = []
    reached_theshold = [False]

    while True:

        # reset the number of counts
        counted_loss_fn = FunctionCounter(loss_fn)

        n_phis = 8
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

            # check if reached threshold
            if np.min(opt.yi) < true_loss:
                reached_theshold[0] = True
                break
        
        # exit if reached threshold
        if reached_theshold[0]:
            n_calls.append(counted_loss_fn.n_calls)
            break

        # get best loss point
        min_idx = np.argmin(opt.yi)
        x0 = opt.Xi[min_idx]

        def callback_fn(intermediate_result):
            # check if reached threshold
            if intermediate_result.fun < true_loss:
                reached_theshold[0] = True
                raise StopIteration

        result = minimize(fun=counted_loss_fn,
                          x0=x0,
                          method="Nelder-Mead",
                          callback=callback_fn,
                          bounds=[(0,2)]*8)
        
        # exit if reached threshold
        if reached_theshold[0]:
            n_calls.append(counted_loss_fn.n_calls)
            break
        
        n_calls.append(counted_loss_fn.n_calls)

    print("n_queries:", n_calls)
    print()

    # save all values
    rf_nm_n_calls.append(n_calls)

    # save results
    with open(path + "seir_8_age_rf_nm_queries_to_true_n_calls.pkl", "wb") as f:
        pickle.dump(rf_nm_n_calls, f)

print("DONE")
