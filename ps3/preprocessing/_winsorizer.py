import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile):
        """
        Initialize the class with given lower quantile and upper quantile
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Compute the lower quantile and upper quantile value based on given dataset
        """ 
        self.lower_quantile_ = np.percentile(X, self.lower_quantile * 100)
        self.upper_quantile_ = np.percentile(X, self.upper_quantile * 100)
        return self.lower_quantile_, self.upper_quantile_

    def transform(self, X):
        """
        Clip the array at the computed quantiles
        """
        return np.clip(X, self.lower_quantile_, self.upper_quantile_)




