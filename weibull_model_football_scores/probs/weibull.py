from scipy.special import gamma
import numpy as np
from functools import lru_cache
from math import gamma
from weibull_model_football_scores import config


@lru_cache(maxsize=None)
def alphas(x: int, j: int, c: float) -> float:
    if x == 0:
        return gamma(c * j + 1) / gamma(j + 1)
    elif j < x:
        raise ValueError(f"{x, j}")
    else:
        return sum(
            [
                alphas(x - 1, m, c) * gamma(c * j - c * m + 1) / gamma(j - m + 1)
                for m in range(x - 1, j)
            ]
        )


def _weibull(x: int, c: float, l: float, t: float = 1, j: int = 0) -> float:
    return (-1) ** (x + j) * (l * t**c) ** j * alphas(x, j, c) / gamma(c * j + 1)


def weibull(x: int, c: float, l: float, t: float = 1) -> float:
    return sum(
        [_weibull(x, c, l, t, j) for j in range(x, x + 50)]
    )  # 50 terms is an approximation of inf, but apparently good enough


def cumulative_weibull(x: int, c: float, l: float, t: float = 1) -> float:
    return sum([weibull(i, c, l, t) for i in range(0, x + 1)])


# k is fic
def frank_copula(u: float, v: float, k: float = config.KAPPA) -> float:
    return (
        -1
        / k
        * np.log(1 + (np.exp(-k * u) - 1) * (np.exp(-k * v) - 1) / (np.exp(-k) - 1))
    )


def weibull_likelihood(
    y1: int, y2: int, l1: float, l2: float, c1=config.C1, c2=config.C2
) -> float:
    x1 = cumulative_weibull(y1, c1, l1)
    x2 = cumulative_weibull(y2, c2, l2)
    x3 = cumulative_weibull(y1 - 1, c1, l1)
    x4 = cumulative_weibull(y2 - 1, c2, l2)

    return (
        frank_copula(x1, x2)
        - frank_copula(x1, x4)
        - frank_copula(x3, x2)
        + frank_copula(x3, x4)
    )
