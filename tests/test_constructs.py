import autograd.numpy as np
from nvxpy import Function


def test_function_initialization():
    def dummy_func(x):
        return x * 2

    func = Function(dummy_func, jac="numerical")
    assert func.func(2) == 4
    assert callable(func.jac)


def test_function_call():
    def dummy_func(x):
        return x + 1

    func = Function(dummy_func, jac="numerical")
    result = func(3)
    assert np.isclose(result.func(3), 4)
    assert callable(result.jac)


def test_numerical_diff():
    def dummy_func(x):
        return x**2

    func = Function(dummy_func, jac="numerical")
    result = func(2)
    numerical_diff = result.jac(dummy_func, 2)[0](1)
    assert callable(result.jac)
    assert np.isclose(numerical_diff, 4.0)


def test_callable_jacobian():
    def dummy_func(x):
        return x * 3

    def custom_jacobian(func, *args):
        return [lambda g: 1.0 * g for _ in args]

    func = Function(dummy_func, jac=custom_jacobian)
    result = func(2)
    assert callable(func.jac)
    assert np.isclose(result.func(2), 6)


def test_invalid_jacobian():
    def dummy_func(x):
        return x * 4

    try:
        Function(dummy_func, jac="invalid")
    except ValueError as e:
        assert str(e) == "Invalid jacobian: invalid"
    else:
        assert False, "ValueError not raised for invalid jacobian"
