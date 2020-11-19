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


import numpy as np


class ObjectiveFunction:
    def __init__(self, dim, x_opt, f_opt, noise_var=0.0, lb=None, ub=None):
        """
        Abstract class that defines the interface for objective functions.

        Parameters
        ----------
        dim : int
            Dimensionality of the input variables.
        x_opt : np.ndarray or list of np.ndarrays, shape(dim,)
            Location(s) of the optimum.
        f_opt : float
            Optimal function value.
        noise_var : float, optional
            variance of the observation noise that is added for each evaluation
        lb : ndarray, shape(dim(x), ), optional
            Lower bound on the optimization parameters.
        ub : ndarray, shape(dim(x), ), optional
            Upper bound on the optimization parameters.
        """
        self._dim = dim
        self._x_opt = x_opt
        self._f_opt = f_opt
        self._noise_var = noise_var
        self._lb = lb
        self._ub = ub

    @property
    def dim(self):
        return self._dim

    @property
    def x_opt(self):
        return self._x_opt

    @property
    def f_opt(self):
        return self._f_opt

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    def __call__(self, x):
        """Evaluate objective function with given noise level."""
        x = np.atleast_2d(x)
        assert x.shape[1] == self._dim
        fx = self._fun(x)
        yx = fx + np.sqrt(self._noise_var) * np.random.randn(*fx.shape)
        assert yx.shape == (x.shape[0], 1)
        return yx

    @staticmethod
    def _fun(x):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, the __call__ method is not implemented."
        )
