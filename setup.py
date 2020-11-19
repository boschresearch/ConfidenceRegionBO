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


from setuptools import setup
from codecs import open
from os import path

name = "crbo"
version = "0.1"
description = "Confidence Region Bayesian Optimization"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the required packages
reqs = ["jupyter", "cython", "numpy", "scipy", "tqdm", "sobol-seq", "GPy>=1.9.9", "matplotlib",
        "stable-baselines", "tensorflow==1.15", "gym", "pyyaml"]

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    install_requires=reqs,
)
