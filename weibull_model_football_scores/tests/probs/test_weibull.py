from weibull_model_football_scores.probs.weibull import weibull, cumulative_weibull
from scipy.stats import poisson
import numpy as np


def test_weibull():
    # c = 1 -> Weibull = Poisson

    X = range(10)
    c = 1
    results = []
    for _lambda in abs(np.random.randn(10)):
        for x in X:
            results.append(
                abs(weibull(x=x, c=c, l=_lambda) - poisson.pmf(x, mu=_lambda))
            )

    assert sum(results) < 10e-4


# def test_cumulative_weibull():
#    cumulative_weibull()
#
