import pytest
import autograd.numpy as np
from nvxpy.atoms.sum import sum
from nvxpy.atoms.maximum import maximum
from nvxpy.atoms.det import det
from nvxpy.atoms.norm import norm
from nvxpy.atoms.polar import PolarDecomposition
from nvxpy.atoms.trace import trace


def test_sum():
    x = np.array([[1, 2], [3, 4]])
    s = sum(x)
    assert np.isclose(s(x), 10)


def test_maximum():
    x = np.array([1, 3, 5])
    y = np.array([2, 2, 6])
    m = maximum(x, y)
    assert np.array_equal(m(x, y), np.array([2, 3, 6]))


def test_det():
    x = np.array([[1, 2], [3, 4]])
    d = det(x)
    assert np.isclose(d(x), -2.0)


def test_norm():
    x = np.array([3, 4])
    n = norm(x)
    assert np.isclose(n(x), 5.0)


def test_polar_decomposition():
    x = np.array([[1, 2], [3, 4]])
    p = PolarDecomposition(x)
    result = p(x)
    assert np.allclose(result @ result.T, np.eye(2))


def test_trace():
    x = np.array([[1, 2], [3, 4]])
    t = trace(x)
    assert np.isclose(t(x), 5)
