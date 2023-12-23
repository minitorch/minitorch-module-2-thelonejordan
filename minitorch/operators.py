"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    raise NotImplementedError("Need to include this file from past assignment.")


def id(x: float) -> float:
    "$f(x) = x$"
    raise NotImplementedError("Need to include this file from past assignment.")


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    raise NotImplementedError("Need to include this file from past assignment.")


def neg(x: float) -> float:
    "$f(x) = -x$"
    raise NotImplementedError("Need to include this file from past assignment.")


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    raise NotImplementedError("Need to include this file from past assignment.")


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    raise NotImplementedError("Need to include this file from past assignment.")


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    raise NotImplementedError("Need to include this file from past assignment.")


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    raise NotImplementedError("Need to include this file from past assignment.")


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    raise NotImplementedError("Need to include this file from past assignment.")


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    raise NotImplementedError("Need to include this file from past assignment.")


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    raise NotImplementedError("Need to include this file from past assignment.")


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    raise NotImplementedError("Need to include this file from past assignment.")


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    raise NotImplementedError("Need to include this file from past assignment.")


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    raise NotImplementedError("Need to include this file from past assignment.")


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    raise NotImplementedError("Need to include this file from past assignment.")


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    raise NotImplementedError("Need to include this file from past assignment.")


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    raise NotImplementedError("Need to include this file from past assignment.")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    raise NotImplementedError("Need to include this file from past assignment.")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    raise NotImplementedError("Need to include this file from past assignment.")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    raise NotImplementedError("Need to include this file from past assignment.")


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    raise NotImplementedError("Need to include this file from past assignment.")
