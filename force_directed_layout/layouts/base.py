import math

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np


class ForceLayoutBase:
    def force(self, alpha: float, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.force(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        return


@dataclass
class _ConstFn:
    x: float

    def __call__(self, *args, **kwargs):
        return self.x


Constant = _ConstFn


def Fn(x: Union[float, Callable[[Any], float]]):
    if callable(x):
        return x
    return _ConstFn(x)


def isnull(x: Optional[float]):
    if x is None:
        return True
    return math.isnan(x)


def jiggle() -> float:
    return (np.random.random() - 0.5) * 1e-6
