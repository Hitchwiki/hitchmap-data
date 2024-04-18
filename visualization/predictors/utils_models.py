import pandas as pd
import numpy as np
from tqdm import tqdm

from map_utils import *


from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator

from numeric_transformers import LogTransformer
from transformed_target_regressor_with_uncertainty import (
    TransformedTargetRegressorWithUncertainty,
)

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# centers data to a zero mean
# use this transformer if you want to center the data outside of the GP model (e.g. for visualization)
class TargetTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, function=(lambda y: y), inverse_function=(lambda y: y)):
        self.function = function
        self.inverse_function = inverse_function
        self.mean = 0

    def fit(self, y):
        self.targets = y
        self.mean = np.mean(self.function(y))

    def transform(self, y):
        return self.function(y) - self.mean

    def inverse_transform(self, y):
        return self.inverse_function(y + self.mean)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


def evaluate(model, train, validation, features=["lon", "lat"]):
    train["pred"] = model.predict(train[features].values)

    print(f"Training RMSE: {root_mean_squared_error(train['wait'], train['pred'])}")
    print(f"Training MAE {mean_absolute_error(train['wait'], train['pred'])}")

    validation["pred"] = model.predict(validation[features].values)

    print(
        f"Validation RMSE: {root_mean_squared_error(validation['wait'], validation['pred'])}"
    )
    print(
        f"Validation MAE {mean_absolute_error(validation['wait'], validation['pred'])}\n"
    )


@ignore_warnings(category=ConvergenceWarning)
def evaluate_cv(estimator, X, y, folds=5):
    cv_result = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=folds,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=True,
        return_estimator=True,
    )

    print(f"Cross-validated averaged metrics...")
    print(
        f"Training RMSE: {cv_result['train_neg_root_mean_squared_error'].mean() * -1}"
    )
    print(f"Training MAE: {cv_result['train_neg_mean_absolute_error'].mean() * -1}")
    print(
        f"Validation RMSE: {cv_result['test_neg_root_mean_squared_error'].mean() * -1}"
    )
    print(f"Validation MAE: {cv_result['test_neg_mean_absolute_error'].mean() * -1}\n")

    # returning one estimators trained on all samples for visualization purposes
    return estimator.fit(X, y)


def get_gpr(initial_kernel):
    gpr = GaussianProcessRegressor(
        kernel=initial_kernel,
        alpha=0.0**2,
        optimizer="fmin_l_bfgs_b",
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=42,
    )

    target_transform_gpr = TransformedTargetRegressorWithUncertainty(
        regressor=gpr, numeric_transformer=LogTransformer()
    )

    return target_transform_gpr


def fit_gpr(gpr, X, y):
    gpr.fit(X, y)

    return gpr


@ignore_warnings(category=ConvergenceWarning)
def fit_gpr_silent(gpr, X, y):
    gpr.fit(X, y)

    return gpr


def get_optimized_gpr(initial_kernel, X, y, verbose=False):
    gpr = get_gpr(initial_kernel=initial_kernel)
    if verbose:
        gpr = fit_gpr(gpr, X, y)
    else:
        gpr = fit_gpr_silent(gpr, X, y)

    return gpr
