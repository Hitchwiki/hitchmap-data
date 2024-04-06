from utils import *
from map_utils import *
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
    ExpSineSquared,
    RationalQuadratic,
)
import pickle
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np
import seaborn as sns
import matplotlib
import shapely
from matplotlib.colors import LogNorm

# log space
points = get_points("../data/points_train.csv")
# for log transformation make sure waiting times are above 0 otherwise we get -inf for those values
points["wait"] = points["wait"].apply(lambda x: 0.1 if x == 0 else x)
region = "europe"
points, polygon, map_boundary = get_points_in_region(points, region)
points["lat"] = points.geometry.y
points["lon"] = points.geometry.x

X = points[["lat", "lon"]].values
y = points["wait"].values

# y_ being the waiting time in the (new) log space
# log space allows us to contrain the prior of functions (in the original space)
# to functions being always positive
# underlying cause: the original ys are not normally distributed (as they do not become less than 0)
y_ = np.log(y)

# assuming mean = 0 in the gp setup
average = np.mean(y_)
y_ = y_ - average
stdv = np.std(y_)

# parameters to optimize

# tuning hyperparameters

l = 1e5
L = [l, l]
sigma = stdv

rbf = RBF(
    length_scale=L, length_scale_bounds=(1e-100, 1e100)
)  # using anisotripic kernel (different length scales for each dimension)
rbf2 = RBF(
    length_scale=L, length_scale_bounds=(1e-100, 1e100)
)  # using anisotripic kernel (different length scales for each dimension)

kernel = (
    ConstantKernel(constant_value=sigma**2, constant_value_bounds=(1e-1, 1e1)) * rbf
    + ConstantKernel(constant_value=sigma**2, constant_value_bounds=(1e-1, 1e1)) * rbf
    + WhiteKernel(noise_level=1.05e0, noise_level_bounds=(1e-1, 1e1))
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.0**2,
    optimizer="fmin_l_bfgs_b",  # maximizing marginal log lik: given the data, how likely is the parameterization of the kernel
    # (if we draw a sample frome the multivariate gaussian with the kernel as covariance matrix, how likely is the data we have seen)
    # prevents overfitting to some degree
    normalize_y=False,
    n_restarts_optimizer=0,
    random_state=42,
)

# have to fit once before using log_marginal_likelihood aka show the data
gp.fit(X, y_)
print(gp.kernel_, np.exp(gp.kernel_.theta))

num = 10
length_scale1 = np.logspace(3, 6, num=num)
length_scale2 = np.logspace(3, 6, num=num)
length_scale1_grid, length_scale2_grid = np.meshgrid(length_scale1, length_scale2)

print("Calculating log marginal likelihood...")
log_marginal_likelihood = [
    gp.log_marginal_likelihood(theta=np.log([scale1, scale1, scale2, scale2]))
    for scale1, scale2 in tqdm(
        zip(length_scale1_grid.ravel(), length_scale2_grid.ravel()),
        total=len(length_scale1_grid.ravel()),
    )
]
log_marginal_likelihood = np.reshape(
    log_marginal_likelihood, newshape=length_scale2_grid.shape
)

print("Plotting...")
vmin, vmax = (-log_marginal_likelihood).min(), (-log_marginal_likelihood).max()
level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=50), decimals=1)
plt.contour(
    length_scale1_grid,
    length_scale2_grid,
    -log_marginal_likelihood,
    levels=level,
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Length-scale for lat")
plt.ylabel("Length-scale for lon")
plt.title("Log-marginal-likelihood")
plt.savefig(f"exp/search.png")
