import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)
    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    lower_quantile_, upper_quantile_ = winsorizer.fit(X)
    
    assert lower_quantile_ == np.percentile(X, lower_quantile * 100), "lower quantile is wrong"
    assert upper_quantile_ == np.percentile(X, upper_quantile * 100), "upper quantile is wrong"

    X_clipped = winsorizer.transform(X)

    assert max(X_clipped) <= upper_quantile_, "upper bound failed"
    assert min(X_clipped) >= lower_quantile_, "lower bound failed"
    assert len(X_clipped) == len(X), "length of array changes"


