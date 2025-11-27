"""Tests for expression.py - operations, shapes, and curvature."""
import autograd.numpy as np
import pytest
import nvxpy as nvx
from nvxpy.variable import Variable
from nvxpy.expression import expr_to_str


class TestExprToStr:
    """Tests for expr_to_str function."""

    def test_expr_to_str_expr(self):
        x = Variable(shape=(2,), name="x")
        expr = x + 1
        result = expr_to_str(expr)
        assert "add" in result

    def test_expr_to_str_array(self):
        arr = np.array([1, 2, 3])
        result = expr_to_str(arr)
        assert "Const" in result
        assert "shape=" in result

    def test_expr_to_str_scalar(self):
        result = expr_to_str(5.0)
        assert "Const" in result
        assert "float" in result


class TestExprRepr:
    """Tests for Expr __repr__ method."""

    def test_repr_binary(self):
        x = Variable(shape=(2,), name="x")
        expr = x + 1
        repr_str = repr(expr)
        assert "add" in repr_str

    def test_repr_unary(self):
        x = Variable(shape=(2,), name="x")
        expr = -x
        repr_str = repr(expr)
        assert "neg" in repr_str

    def test_expr_hash(self):
        x = Variable(shape=(2,), name="x")
        expr1 = x + 1
        expr2 = x + 1
        # Different objects should still be hashable
        hash1 = hash(expr1)
        hash2 = hash(expr2)
        assert isinstance(hash1, int)
        assert isinstance(hash2, int)


class TestExprValue:
    """Tests for Expr.value property."""

    def test_value_simple(self):
        x = Variable(shape=(2,), name="x")
        x.value = np.array([1.0, 2.0])
        expr = x + 1
        result = expr.value
        assert np.allclose(result, np.array([2.0, 3.0]))

    def test_value_nested(self):
        x = Variable(shape=(2,), name="x")
        x.value = np.array([1.0, 2.0])
        expr = (x + 1) * 2
        result = expr.value
        assert np.allclose(result, np.array([4.0, 6.0]))


class TestExprOperations:
    """Tests for various expression operations."""

    def test_transpose(self):
        X = Variable(shape=(2, 3), name="X")
        expr = X.T
        assert expr.op == "transpose"
        assert expr.shape == (3, 2)

    def test_flatten(self):
        X = Variable(shape=(2, 3), name="X")
        expr = X.flatten()
        assert expr.op == "flatten"
        assert expr.shape == (6,)

    def test_radd(self):
        x = Variable(shape=(2,), name="x")
        expr = 1 + x
        assert expr.op == "add"

    def test_rsub(self):
        x = Variable(shape=(2,), name="x")
        expr = 1 - x
        assert expr.op == "sub"

    def test_rmul(self):
        x = Variable(shape=(2,), name="x")
        expr = 2 * x
        assert expr.op == "mul"

    def test_rmatmul(self):
        X = Variable(shape=(2, 3), name="X")
        A = np.ones((4, 2))
        expr = A @ X
        assert expr.op == "matmul"
        assert expr.shape == (4, 3)

    def test_rtruediv(self):
        x = Variable(shape=(2,), name="x")
        expr = 1 / x
        assert expr.op == "div"

    def test_getitem_int(self):
        x = Variable(shape=(5,), name="x")
        expr = x[2]
        assert expr.op == "getitem"
        assert expr.shape == (1,)

    def test_getitem_slice(self):
        x = Variable(shape=(5,), name="x")
        expr = x[1:4]
        assert expr.op == "getitem"
        assert expr.shape == (3,)

    def test_getitem_tuple(self):
        X = Variable(shape=(3, 4), name="X")
        expr = X[1, :]
        assert expr.op == "getitem"

    def test_rshift_constraint(self):
        X = Variable(shape=(3, 3), name="X")
        cons = X >> 0
        assert cons.op == ">>"

    def test_lshift_constraint(self):
        X = Variable(shape=(3, 3), name="X")
        cons = X << 0
        assert cons.op == "<<"

    def test_xor_discrete_set(self):
        x = Variable(shape=(2,), name="x")
        cons = x ^ [1, 2, 3]
        assert cons.op == "in"


class TestExprCurvature:
    """Tests for curvature inference."""

    def test_add_affine_affine(self):
        x = Variable(shape=(2,), name="x")
        y = Variable(shape=(2,), name="y")
        expr = x + y
        assert expr.curvature == nvx.AFFINE

    def test_add_affine_constant(self):
        x = Variable(shape=(2,), name="x")
        expr = x + 1
        assert expr.curvature == nvx.AFFINE

    def test_add_convex_convex(self):
        x = Variable(shape=(2,), name="x")
        expr = x ** 2 + x ** 2
        assert expr.curvature == nvx.CONVEX

    def test_sub_affine_affine(self):
        x = Variable(shape=(2,), name="x")
        y = Variable(shape=(2,), name="y")
        expr = x - y
        assert expr.curvature == nvx.AFFINE

    def test_mul_positive_constant(self):
        x = Variable(shape=(2,), name="x")
        expr = 2 * x
        assert expr.curvature == nvx.AFFINE

    def test_mul_negative_constant(self):
        x = Variable(shape=(2,), name="x")
        expr = -2 * x
        assert expr.curvature == nvx.AFFINE

    def test_mul_negative_convex(self):
        x = Variable(shape=(2,), name="x")
        expr = -1 * (x ** 2)
        assert expr.curvature == nvx.CONCAVE

    def test_div_positive_constant(self):
        x = Variable(shape=(2,), name="x")
        expr = x / 2
        assert expr.curvature == nvx.AFFINE

    def test_div_negative_constant(self):
        x = Variable(shape=(2,), name="x")
        expr = x / (-2)
        assert expr.curvature == nvx.AFFINE

    def test_pow_squared(self):
        x = Variable(shape=(2,), name="x")
        expr = x ** 2
        assert expr.curvature == nvx.CONVEX

    def test_pow_one(self):
        x = Variable(shape=(2,), name="x")
        expr = x ** 1
        assert expr.curvature == nvx.AFFINE

    def test_neg(self):
        x = Variable(shape=(2,), name="x")
        expr = -x
        assert expr.curvature == nvx.AFFINE

    def test_neg_convex(self):
        x = Variable(shape=(2,), name="x")
        expr = -(x ** 2)
        assert expr.curvature == nvx.CONCAVE

    def test_matmul_affine_constant(self):
        X = Variable(shape=(2, 3), name="X")
        A = np.ones((3, 4))
        expr = X @ A
        assert expr.curvature == nvx.AFFINE

    def test_matmul_constant_affine(self):
        X = Variable(shape=(2, 3), name="X")
        A = np.ones((4, 2))
        expr = A @ X
        assert expr.curvature == nvx.AFFINE


class TestExprShape:
    """Tests for shape inference."""

    def test_add_broadcast_scalar(self):
        x = Variable(shape=(2, 3), name="x")
        expr = x + 1
        assert expr.shape == (2, 3)

    def test_add_broadcast_row(self):
        x = Variable(shape=(2, 3), name="x")
        row = np.ones((1, 3))
        expr = x + row
        assert expr.shape == (2, 3)

    def test_mul_broadcast(self):
        x = Variable(shape=(2, 3), name="x")
        expr = x * 2
        assert expr.shape == (2, 3)

    def test_matmul_matrix_matrix(self):
        X = Variable(shape=(2, 3), name="X")
        Y = Variable(shape=(3, 4), name="Y")
        expr = X @ Y
        assert expr.shape == (2, 4)

    def test_matmul_vector_vector(self):
        x = Variable(shape=(3,), name="x")
        y = Variable(shape=(3,), name="y")
        expr = x @ y
        assert expr.shape == ()

    def test_matmul_matrix_vector(self):
        X = Variable(shape=(2, 3), name="X")
        y = Variable(shape=(3,), name="y")
        expr = X @ y
        # Shape may include trailing 1 depending on implementation
        assert expr.shape in ((2,), (2, 1))

    def test_pow_shape(self):
        x = Variable(shape=(2, 3), name="x")
        expr = x ** 2
        assert expr.shape == (2, 3)

    def test_neg_shape(self):
        x = Variable(shape=(2, 3), name="x")
        expr = -x
        assert expr.shape == (2, 3)

    def test_getitem_row(self):
        X = Variable(shape=(3, 4), name="X")
        expr = X[0]
        assert expr.shape == (4,)

    def test_getitem_slice_rows(self):
        X = Variable(shape=(5, 4), name="X")
        expr = X[1:3]
        assert expr.shape == (2, 4)

    def test_shape_incompatible_raises(self):
        x = Variable(shape=(2,), name="x")
        y = Variable(shape=(3,), name="y")
        expr = x + y
        with pytest.raises(ValueError):
            _ = expr.shape

    def test_matmul_incompatible_raises(self):
        X = Variable(shape=(2, 3), name="X")
        Y = Variable(shape=(4, 5), name="Y")
        expr = X @ Y
        with pytest.raises(ValueError):
            _ = expr.shape
