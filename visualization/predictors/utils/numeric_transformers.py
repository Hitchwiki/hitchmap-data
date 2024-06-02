# from https://github.com/scikit-learn/scikit-learn/issues/24638

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin


# see https://en.wikipedia.org/wiki/Log-normal_distribution#Arithmetic_moments
# needed for Target transformer when we have std
def inverse_log_mean(log_mean, log_std):
    """
    Calculate the mean in the original scale for a log-normal distribution.

    This function transforms the mean from the logarithmic scale back to the
    original scale. It corrects for the skewness of the log-normal distribution
    by incorporating the variance.
    """
    log_var = log_std**2
    return np.exp(log_mean + log_var / 2) - 1e-7


def inverse_log_std(log_mean, log_std):
    """
    Calculate the standard deviation in the original scale for a log-normal distribution.

    This function transforms the standard deviation from the logarithmic scale
    back to the original scale. It leverages the properties of the log-normal
    distribution, using the exponential transformation to compute the variance
    and then obtaining the standard deviation.
    """
    log_var = log_std**2
    var = np.exp(2 * log_mean + log_var) * (np.exp(log_var) - 1)
    return np.sqrt(var)


def inverse_sqrt_mean(sqrt_mean, sqrt_std):
    """
    This function calculates the mean in the original space.

    The mean of squared values isn't just the square of the mean of the values
    because squaring is a non-linear operation. A reasonable approximation is
    to use the second moment about the mean (i.e., the variance plus the square
    of the mean) of the sqrt-transformed values.
    """
    return sqrt_mean**2 + sqrt_std**2


def inverse_sqrt_std(sqrt_mean, sqrt_std):
    """
    This function calculates the standard deviation in the original space.

    We need to consider how variance (the square of the standard deviation)
    transforms. The variance of the squared values can be approximated as the
    fourth moment about the mean minus the square of the second moment about
    the mean.
    """
    return np.sqrt(4 * sqrt_mean**2 * sqrt_std**2 + 2 * sqrt_std**4)


def log_plus_tiny(x):
    """We add a tiny value to allow zero inputs"""
    return np.log(x + 1e-7)


def exp_minus_tiny(x):
    """To invert log_plus_tiny"""
    return np.exp(x) - 1e-7


class Transformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        description="",
        func=lambda x: x,
        inverse_func=lambda x: x,
        allows_negatives=False,
        inverse_mean_func=None,
        inverse_std_func=None,
    ):
        self.description = description
        self.func = func
        self.inverse_func = inverse_func
        self.allows_negatives = allows_negatives
        self.inverse_mean_func = inverse_mean_func
        self.inverse_std_func = inverse_std_func


class LogTransformer(Transformer):
    def __init__(
        self,
        description="log(x + 1)",
        func=log_plus_tiny,
        inverse_func=exp_minus_tiny,
        allows_negatives=False,
        inverse_mean_func=inverse_log_mean,
        inverse_std_func=inverse_log_std,
    ):
        self.description = description
        self.func = func
        self.inverse_func = inverse_func
        self.allows_negatives = allows_negatives
        self.inverse_mean_func = inverse_mean_func
        self.inverse_std_func = inverse_std_func

def f(x): return np.log(x + 1)
def g(x): return np.exp(x) - 1
def h(x, y): return np.exp(x) - 1
def i(x, y): return y
# the correct stdvs (uncertainties) are not important - relative values matter more
# the correct back-transformation but yields too high values for our use case
# np.log(x + 1e-7) should be used but yielded worse results in our use case
# to be investigated
class MyLogTransformer(Transformer):
    def __init__(
        self,
        description="log(x + 1e-7)",
        func=f,
        inverse_func=g,
        allows_negatives=False,
        inverse_mean_func=h,
        inverse_std_func=i,
    ):
        self.description = description
        self.func = func
        self.inverse_func = inverse_func
        self.allows_negatives = allows_negatives
        self.inverse_mean_func = inverse_mean_func
        self.inverse_std_func = inverse_std_func

class SqrtTransformer(Transformer):
    def __init__(self):
        super().__init__(
            description="sqrt(x)",
            func=np.sqrt,
            inverse_func=np.square,
            allows_negatives=False,
            inverse_mean_func=inverse_sqrt_mean,
            inverse_std_func=inverse_sqrt_std,
        )


class IdentityTransformer(Transformer):
    def __init__(self):
        super().__init__(
            description="",
            func=lambda x: x,
            inverse_func=lambda x: x,
            allows_negatives=True,
        )


# ⚠️ If you change or remove value add a mapping in converters.py
# Values defined here are used in the UI & API dropdowns
class NumericTransformerOption(Enum):
    na = "No transformation"
    log = "Log"
    sqrt = "Square root"

    def get_transformer(self) -> Transformer:
        if self == NumericTransformerOption.log:
            return LogTransformer()
        elif self == NumericTransformerOption.sqrt:
            return SqrtTransformer()
        else:
            return IdentityTransformer()


def get_transformer_with_least_skew(
    y: list[float | int | None] | pd.Series | np.ndarray,
) -> NumericTransformerOption:
    """
    Apply each available transformation to the data and return the one with the
    lowest abs skew.
    """
    try:
        y_array = np.array(y, dtype=float)
    except ValueError:
        return NumericTransformerOption.na

    # Filter out NaN values
    y_array = y_array[~np.isnan(y_array)]

    # Handle the case where all values are NaN or the array is empty
    if y_array.size == 0:
        return NumericTransformerOption.na

    any_negative = any(y_array < 0)
    min_skew = abs(skew(y_array))
    best_transformation = NumericTransformerOption.na

    # try each enum value that has a transformation function
    for t in NumericTransformerOption:
        transformer = t.get_transformer()
        if isinstance(transformer, IdentityTransformer):
            continue

        if any_negative and not transformer.allows_negatives:
            continue

        transformed_y = transformer.func(y_array)
        transformed_skew = abs(skew(transformed_y))
        print(f"Skew for {t}: {transformed_skew}")

        if transformed_skew < min_skew:
            min_skew = transformed_skew
            best_transformation = t

    return best_transformation
