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

    def __init__(self, regressor: BaseEstimator, numeric_transformer: Transformer):
        super().__init__()
        self.numeric_transformer = numeric_transformer
        self.regressor = regressor

    def fit(self, X, y, **fit_params):
        """
        Fit the regressor and the transformer.
        """
        # setting parameters required for fitting
        if (
            self.numeric_transformer.inverse_mean_func is None
            or self.numeric_transformer.inverse_std_func is None
        ):
            raise ValueError(
                "To support predictions with a standard deviation a transformer"
                "must have inverse_mean_func and inverse_std_func functions."
            )

        self.func = self.numeric_transformer.func
        self.inverse_func = self.numeric_transformer.inverse_func

        return super().fit(X, y, **fit_params)

    def predict(self, X, return_std=False, transform_predictions=True, verbose=True, **predict_params):
        """
        Predict using the underlying regressor and transform the result back.
        """
        print(f"Model called for prediction with X of shape {X.shape}")
        # always return the standard deviation as it is required for the proper inverse_transform
        # regressor_ is the fitted regressor
        model: BaseEstimator = self.regressor_
        tran_pred, tran_std = model.predict(X, return_std=True)
        if transform_predictions:
            tran_pred = self.numeric_transformer.inverse_mean_func(tran_pred, tran_std)
        pred = tran_pred
        std = self.numeric_transformer.inverse_std_func(tran_pred, tran_std)
        if return_std:
            return pred, std
        else:
            return pred
