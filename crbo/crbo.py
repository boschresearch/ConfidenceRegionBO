#  Confidence Region Bayesian Optimization -- Reference Implementation
#  Copyright (c) 2020 Robert Bosch GmbH
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
# 
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import json
import GPy
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.optimize as sciopt
import scipy.stats as scistats
import sobol_seq
import time

from copy import deepcopy
from tqdm import tqdm

from crbo.acquisition_functions import LowerConfidenceBound, ThompsonSampling
from crbo.constraints import ConfidenceRegionConstraint
from crbo import sampling
from crbo import util


class BayesianOptimization(object):
    def __init__(
        self,
        fun,
        max_iter,
        bounds,
        model=None,
        acq_type="lcb",
        optimizer_strategy="sample_local",
        optimize_hypers=False,
        init_hypers=None,
        n_init=None,
        x_init=None,
        y_init=None,
        normalizer_type="neutral",
        verbosity=False,
        fun_args=(),
    ):
        self.__name__ = "bo-base"
        assert isinstance(bounds, sciopt.Bounds)
        self.bounds = bounds

        # Always map input from unit cube to respective bounds when evaluating the objective
        self._fun = fun
        self._fun_args = fun_args
        self.f = lambda x: self._fun(util.denormalize_input(x, self.bounds), *self._fun_args)

        self.max_iter = max_iter  # Maximum number of BO iterations
        self.n_init = n_init  # Number of initial design points
        self.x_init = x_init  # Initial design points
        self.y_init = y_init  #
        self.normalizer_type = normalizer_type
        self.optimizer_strategy = optimizer_strategy
        self.optimize_hypers = optimize_hypers
        self.init_hypers = init_hypers
        self.verbosity = verbosity

        # Sample points are used for optimization of the acquisition function
        if self.optimizer_strategy == "sample_only":
            self._n_max_samples = min(200 * self._fun.dim, 20000)  # Maximum number of samples
        elif self.optimizer_strategy == "sample_local":
            self._n_max_samples = 1000
        self._sample_points = None

        # Store the time it takes to optimize acquisition function and update model
        self._iter_duration = []

        # Choose acquisition function
        if acq_type == "lcb":
            self.acq_fun = LowerConfidenceBound(beta=2.0)
        elif acq_type == "ts":
            self.acq_fun = ThompsonSampling()
            if optimizer_strategy == "sample_local":
                raise ValueError("TS on grid not compatible with local optimizer.")
        else:
            raise ValueError()

        # Build initial GP model 
        self.x_init = self._initial_design()
        self.y_init = self.f(self.x_init)
        self.y_init, self.norm_mu, self.norm_std = util.normalize_output(
            self.y_init, mode=self.normalizer_type
        )

        # Build model (noise and signal variance are scaled)
        kernel = GPy.kern.Matern52(
            input_dim=self._fun.dim,
            lengthscale=init_hypers["lengthscale"],
            variance=init_hypers["variance"] / self.norm_std ** 2,
        )

        # The likelihood noise has to be scaled with the scaling of the data
        noise_var = init_hypers["noise_var"] / self.norm_std ** 2
        self.model = GPy.models.GPRegression(
            self.x_init, self.y_init, kernel=kernel, noise_var=noise_var
        )

        # Set priors on hyperparameters (emphasizing high noise to signal ratio)
        if self.optimize_hypers:
            self.model.kern.lengthscale.set_prior(GPy.priors.Gamma(3.0, 6.0))
            self.model.kern.variance.set_prior(GPy.priors.Gamma(2.0, 0.15))
            self.model.likelihood.variance.set_prior(GPy.priors.Gamma(1.1, 0.05))


    def run(self):
        # Minimalistic BO loop
        for i_iter in tqdm(range(self.max_iter), disable=not self.verbosity):
            # Suggest new point by optimizing acquisition function
            t0 = time.time()
            x_new = self._next_point()
            self._iter_duration.append(time.time() - t0)

            y_new = self.f(x_new)
            self._update_model(x_new, y_new)  # At this point, y_new is NOT normalized

        self.res_dict = self._create_results_dictionary()
        return self.model, self.res_dict

    def save_results(self, res_dir="./", run_id=0):
        """Save the results in a .json file."""
        results_path = os.path.join(res_dir, f"{run_id:03d}_results.json")
        json.dump(self.res_dict, open(results_path, "w"), sort_keys=True, indent=2)

    def _create_results_dictionary(self):
        res_dict = {
            "x_eval": util.denormalize_input(self.model.X, self.bounds).tolist(),
            "y_eval": util.denormalize_output(self.model.Y, self.norm_mu, self.norm_std).tolist(),
            "opt_type": self.__name__,
            "acq_type": self.acq_fun.__name__,
            "objective": self._fun.__name__,
            "gp_model": self.model.to_dict(),
            "normalizer_type": self.normalizer_type,
            "bounds": {"lb": self.bounds.lb.tolist(), "ub": self.bounds.ub.tolist()},
            "iter_duration": self._iter_duration,
        }
        return res_dict

    def _update_model(self, x_new, y_new):
        x_new = np.atleast_2d(x_new)
        y_new = np.atleast_2d(y_new)
        assert x_new.shape[0] == y_new.shape[0]
        X_norm = np.vstack((self.model.X, x_new))
        Y = util.denormalize_output(self.model.Y, self.norm_mu, self.norm_std)
        Y = np.vstack((Y, y_new))
        Y_norm, self.norm_mu, self.norm_std = util.normalize_output(Y, mode=self.normalizer_type)
        self.model.set_XY(X_norm, Y_norm)
        if self.optimize_hypers:
            self.model.optimize()
        elif self.init_hypers:
            signal_var = self.init_hypers["variance"] / self.norm_std ** 2
            self.model.kern.variance = signal_var

            noise_var = self.init_hypers["noise_var"] / self.norm_std ** 2
            self.model.likelihood.variance[:] = noise_var

        ell = np.clip(self.model.kern.lengthscale.values, 0.05, 0.5)
        self.model.kern.lengthscale[:] = ell

    def _next_point(self):
        self.acq_fun.setup(self.model)
        if self._sample_points is None:
            self._sample_points = self._space_filling_design(n=self._n_max_samples)
        else:
            self._sample_points = self._resample()

        fs = self.acq_fun(self._sample_points, self.model)
        x_opt = self._sample_points[np.argmin(fs)]

        # Optionally, start local, gradient-basd optimizer from best point
        if self.optimizer_strategy == "sample_local":
            x_opt = self._local_opt(x_opt)[0]

        return x_opt

    def _initial_design(self):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )

    def _space_filling_design(self, n):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )

    def _resample(self):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )

    def _local_opt(self, x0):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )


class ConfidenceRegionBayesianOptimization(BayesianOptimization):
    """Confidence Region BO as proposed in our paper."""
    def __init__(
        self,
        fun,
        max_iter,
        bounds,
        gamma,
        model=None,
        acq_type="lcb",
        optimizer_strategy="sample_local",
        optimize_hypers=False,
        init_hypers=None,
        n_init=None,
        x_init=None,
        y_init=None,
        normalizer_type="neutral",
        verbosity=False,
        fun_args=(),
    ):
        self.conf_region = ConfidenceRegionConstraint(gamma)
        self.__name__ = "crbo"

        super().__init__(
            fun,
            max_iter,
            bounds=bounds,
            model=model,
            acq_type=acq_type,
            optimizer_strategy=optimizer_strategy,
            optimize_hypers=optimize_hypers,
            init_hypers=init_hypers,
            n_init=n_init,
            x_init=x_init,
            y_init=y_init,
            normalizer_type=normalizer_type,
            verbosity=verbosity,
            fun_args=fun_args,
        )

    def _create_results_dictionary(self):
        res_dict = super()._create_results_dictionary()
        res_dict["gamma"] = self.conf_region.gamma
        return res_dict

    def _initial_design(self):
        """Given the initial data point, sample n_init - 1 more points around it. Specifically,
        if only one point is given, the confidence region is given by a Ball. For the SE kernel
        we can actually compute the radius of this ball analytically and we then sample the 
        initial points on the surface of said ball."""
        if self.x_init is None:
            raise ValueError("CRBO requires exactly one initial data point.")
        if (self.x_init is not None) and (self.x_init.shape[0] > 1):
            raise ValueError("CRBO requires exactly one initial data point.")
        if self.n_init is None:
            print("You did not specify how many initial data points should be used. It is highly")
            print("recommended to use more than only one.")
            x0 = self.x_init
        else:
            print(f"You provided 1 initial data point but required {self.n_init}.")
            print(f"Generating {self.n_init - 1} more points.")
            # Assuming that we have a RBF kernel and an initial lengthscale `ell`, compute
            # radius of the confidence region (for one data points it's just a sphere).
            ell = 0.2  
            r0 = ell * np.sqrt(-np.log(1 - self.conf_region.gamma ** 2))
            x0 = np.random.randn(self.n_init - 1, self._fun.dim)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            x0 *= r0

            x_init_normalized = util.normalize_input(self.x_init, self.bounds)
            x0 += x_init_normalized
            x0 = np.vstack((x_init_normalized, x0))

        x0 = np.clip(x0, 0.0, 1.0)
        return x0

    def _space_filling_design(self, n):
        """Use LevelsetSubspace (or Hit-and-Run) sampler to uniformly fill the confidence region
        with sample points. For improved coverage, we start multiple sample chains starting from
        the data points of the current GP model. Since the sampler makes use of parallel function
        evaluations, it is recommended to have multiple chains even if only few data ponints exist.
        We'll therefore randomly sample multiple data points with replacement.
        Note that the confidence region actually might be empty. This happens sometimes when 1) the
        the confidence region is chosen rather small (like 0.3) or 2) in the beginning of the
        optimization, when the noise variance is quite large with respect to to the signal variance. 
        To deal with this issue, we inflate the signal variance until the confidence region
        constraint is fulfilled for all data points of the GP. """
        n_increases, max_increases = 0, 50
        while self.conf_region.isempty(self.model) and n_increases < max_increases:
            # Important to fix the model when setting the variance by hand
            self.model.update_model(False)
            self.model.kern.variance.values[:] *= 1.2
            self.model.update_model(True)
            n_increases += 1
            if n_increases == max_increases:
                print(self.model)
                raise RuntimeError(
                    "Confidence region still empty after the maximum amount of increases of the confidence region."
                )

        sampler = sampling.LevelsetSubspaceSampler(
            fun=self.conf_region, fun_args=(self.model,), w=0.1
        )
        x0 = self.model.X[np.random.choice(self.model.X.shape[0], size=(256,), replace=True)]
        x_samples = sampler(n, x0)
        unit_bounds = sciopt.Bounds(lb=np.zeros((self._fun.dim,)), ub=np.ones((self._fun.dim,)))
        x_samples = util.project_into_bounds(x_samples, unit_bounds)
        return x_samples

    def _resample(self):
        """Due to the (possibly) changing design space, the percentage of re-sampled points
        for CRBO is larger in comparison to BoxBO. Further, we need to be careful that old
        sample points are still within the confidence region which can happen when use hyper
        parameters are inferred and the lengthscale decreases."""

        # When re-sampling, we need to check for samples that are now out of the conf region
        ind_out = np.where(self.conf_region(self._sample_points, self.model) <= 0)[0]
        n_new_samples = max(ind_out.shape[0], self._n_max_samples // 2)

        # Re-sample points that are now out of bounds and some new ones
        xs = self._space_filling_design(n=n_new_samples)
        ind_rand = np.random.choice(
            self._n_max_samples, (n_new_samples - ind_out.shape[0],), replace=False
        )
        ind_rand = np.concatenate((ind_out, ind_rand)) if ind_rand.size else ind_out
        self._sample_points[ind_rand] = xs
        return self._sample_points

    def _local_opt(self, x0):
        """Around the initial guess x0, find local optimum of the acquisition function given the
        confidence region consraint and the outer unit box consraint."""
        constraints_scipy = {
            "type": "ineq",
            "fun": self.conf_region,
            "jac": self.conf_region.jac,
            "args": (self.model,),
        }
        options = {"maxiter": 100}
        tol = 1e-4
        acq_args = (self.model, True)

        # Put small box in order to force the optimizer to stay close to initial guess
        lb = np.clip(x0 - 0.5 * self.model.kern.lengthscale, 0.0, +np.inf)
        ub = np.clip(x0 + 0.5 * self.model.kern.lengthscale, -np.inf, 1.0)
        small_bounds = sciopt.Bounds(lb, ub)

        res = sciopt.minimize(
            self.acq_fun,
            x0,
            args=acq_args,
            bounds=small_bounds,
            jac=acq_args[1],
            constraints=constraints_scipy,
            tol=tol,
            options=options,
        )

        f0 = self.acq_fun(x0, self.model)

        if res["fun"] < f0 and res["success"]:
            return res["x"], res["fun"]
        if res["fun"] > f0 and res["success"]:
            return x0, f0
        elif res["success"] and res["nit"] == 1:
            return res["x"], res["fun"]
        elif res["status"] == 9 and res["fun"] < f0:
            return res["x"], res["fun"]
        else:
            return x0, f0
