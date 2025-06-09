import autograd.numpy as np
from nvxpy import Function


def test_function_initialization():
    def dummy_func(x):
        return x * 2

    func = Function(dummy_func, left=None, jac="numerical")
    assert func.func(2) == 4
    assert callable(func.jac)


def test_function_call():
    def dummy_func(x):
        return x + 1

    func = Function(dummy_func, left=None)
    assert np.isclose(func(3), 4)


def test_numerical_diff():
    def dummy_func(x):
        return x**2

    func = Function(dummy_func, left=None)
    numerical_diff = func._numerical_diff(None, 2)
    assert callable(numerical_diff)
    # Test the numerical differentiation
    assert np.isclose(numerical_diff(1), 4.0)
