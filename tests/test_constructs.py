import autograd.numpy as np
import pytest
from nvxpy import Function
from nvxpy.variable import Variable
from nvxpy.constants import Curvature


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


def test_function_expr_repr():
    """Test FunctionExpr __repr__ method."""

    def dummy_func(x):
        return x**2

    func = Function(dummy_func, jac="numerical")
    x = Variable(shape=(2,), name="x")
    expr = func(x)

    # repr should include 'func' and argument info
    repr_str = repr(expr)
    assert "func" in repr_str


def test_function_expr_hash():
    """Test FunctionExpr __hash__ method."""

    def dummy_func(x):
        return x**2

    func = Function(dummy_func, jac="numerical")
    x = Variable(shape=(2,), name="x")
    expr1 = func(x)
    expr2 = func(x)

    # Both should be hashable
    hash1 = hash(expr1)
    hash2 = hash(expr2)
    assert isinstance(hash1, int)
    assert isinstance(hash2, int)


def test_function_expr_value():
    """Test FunctionExpr value property."""

    def square_func(x):
        return x**2

    func = Function(square_func, jac="numerical")
    x = Variable(shape=(2,), name="x")
    x.value = np.array([3.0, 4.0])

    expr = func(x)
    result = expr.value
    assert np.allclose(result, np.array([9.0, 16.0]))


def test_function_expr_curvature():
    """Test FunctionExpr curvature property."""

    def dummy_func(x):
        return x**2

    func = Function(dummy_func, jac="numerical")
    x = Variable(shape=(2,), name="x")
    expr = func(x)

    # User-defined functions have UNKNOWN curvature
    assert expr.curvature == Curvature.UNKNOWN


def test_function_expr_shape_from_kwarg():
    """Test FunctionExpr shape from Function shape kwarg."""

    def dummy_func(x):
        return x**2

    func = Function(dummy_func, jac="numerical", shape=(2,))
    x = Variable(shape=(2,), name="x")
    expr = func(x)

    assert expr.shape == (2,)


def test_function_expr_shape_inference():
    """Test FunctionExpr shape inference from evaluation."""

    def dummy_func(x):
        return np.array([x[0], x[1], x[0] + x[1]])

    func = Function(dummy_func, jac="numerical")
    x = Variable(shape=(2,), name="x")
    x.value = np.array([1.0, 2.0])

    expr = func(x)
    assert expr.shape == (3,)


def test_function_kwarg_variable_error():
    """Test that passing Variable as kwarg raises error."""

    def dummy_func(x, scale=1.0):
        return x * scale

    func = Function(dummy_func, jac="numerical")
    x = Variable(shape=(2,), name="x")
    y = Variable(shape=(2,), name="y")

    with pytest.raises(TypeError, match="keyword arguments"):
        func(x, scale=y)
