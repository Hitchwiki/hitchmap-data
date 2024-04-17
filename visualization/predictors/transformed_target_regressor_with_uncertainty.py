# from https://github.com/scikit-learn/scikit-learn/issues/24638


from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from numeric_transformers import Transformer

class TransformedTargetRegressorWithUncertainty(TransformedTargetRegressor):
    """
    Thin wrapper over sklearn.compose.TransformedTargetRegressor.
    Allows for predict methods that return a tuple of (mean, std) instead of just mean.
    Also allows for corrected inverse_transform methods that take into account the
    target distribution transformation.
    """

    def __init__(self, regressor: BaseEstimator, transformer: Transformer):
        if (
            transformer.inverse_mean_func is None
            or transformer.inverse_std_func is None
        ):
            raise ValueError(
                "To support predictions with a standard deviation a transformer"
                "must have inverse_mean_func and inverse_std_func functions."
            )
        self.inverse_mean_func = transformer.inverse_mean_func
        self.inverse_std_func = transformer.inverse_std_func
        super().__init__(
            regressor, func=transformer.func, inverse_func=transformer.inverse_func
        )

    def predict(self, X, return_std=False, **predict_params):
        """
        Predict using the underlying regressor and transform the result back.
        """
        if return_std:
            model: BaseEstimator = self.regressor_
            tran_pred, tran_std = model.predict(X, return_std=return_std)
            pred = self.inverse_mean_func(tran_pred, tran_std)
            std = self.inverse_std_func(tran_pred, tran_std)
            return pred, std
        return super().predict(X, **predict_params)
