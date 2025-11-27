import autograd.numpy as np
import nvxpy as nvx
from nvxpy.atoms.sum import sum
from nvxpy.atoms.maximum import maximum
from nvxpy.atoms.det import det
from nvxpy.atoms.norm import norm
from nvxpy.atoms.polar import PolarDecomposition
from nvxpy.atoms.trace import trace
from nvxpy.atoms.abs import abs as nvx_abs
from nvxpy.atoms.amax import amax
from nvxpy.atoms.amin import amin
from nvxpy.atoms.logdet import logdet
from nvxpy.variable import Variable


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


def test_abs():
    """Test abs atom."""
    x = np.array([-3, 4, -5])
    a = nvx_abs(x)
    assert np.array_equal(a(x), np.array([3, 4, 5]))


def test_abs_curvature():
    """Test abs curvature analysis."""
    x = Variable(shape=(3,), name="x")
    a = nvx_abs(x)
    # abs of affine is convex
    assert a.curvature == nvx.CONVEX

    # abs of constant is constant
    const = np.array([1, 2, 3])
    a_const = nvx_abs(const)
    assert a_const.curvature == nvx.CONSTANT


def test_amax():
    """Test amax atom."""
    x = np.array([1, 5, 3])
    a = amax(x)
    assert a(x) == 5


def test_amax_curvature():
    """Test amax curvature analysis."""
    x = Variable(shape=(3,), name="x")
    a = amax(x)
    # amax of affine is convex
    assert a.curvature == nvx.CONVEX

    # amax of convex is convex
    x_sq = x ** 2
    a_sq = amax(x_sq)
    assert a_sq.curvature == nvx.CONVEX

    # amax of constant is constant
    const = np.array([1, 2, 3])
    a_const = amax(const)
    assert a_const.curvature == nvx.CONSTANT


def test_amin():
    """Test amin atom."""
    x = np.array([1, 5, 3])
    a = amin(x)
    assert a(x) == 1


def test_amin_curvature():
    """Test amin curvature analysis."""
    x = Variable(shape=(3,), name="x")
    a = amin(x)
    # amin of affine is concave
    assert a.curvature == nvx.CONCAVE

    # amin of concave is concave
    neg_x = -x
    a_neg = amin(neg_x)
    assert a_neg.curvature == nvx.CONCAVE

    # amin of constant is constant
    const = np.array([1, 2, 3])
    a_const = amin(const)
    assert a_const.curvature == nvx.CONSTANT


def test_logdet():
    """Test logdet atom."""
    x = np.array([[2, 0], [0, 3]])
    ld = logdet(x)
    assert np.isclose(ld(x), np.log(6.0))


def test_norm_frobenius():
    """Test Frobenius norm."""
    x = np.array([[1, 2], [3, 4]])
    n = norm(x, ord="fro")
    expected = np.sqrt(1 + 4 + 9 + 16)
    assert np.isclose(n(x), expected)


def test_norm_curvature():
    """Test norm curvature."""
    x = Variable(shape=(3,), name="x")
    n = norm(x)
    # norm of affine is convex
    assert n.curvature == nvx.CONVEX


def test_maximum_curvature():
    """Test maximum curvature."""
    x = Variable(shape=(3,), name="x")
    y = Variable(shape=(3,), name="y")
    m = maximum(x, y)
    # maximum of two affine is convex
    assert m.curvature == nvx.CONVEX


def test_sum_curvature():
    """Test sum curvature."""
    x = Variable(shape=(3,), name="x")
    s = sum(x)
    # sum of affine is affine
    assert s.curvature == nvx.AFFINE


def test_trace_curvature():
    """Test trace curvature."""
    X = Variable(shape=(3, 3), name="X")
    t = trace(X)
    # trace of affine is affine
    assert t.curvature == nvx.AFFINE


def test_det_curvature():
    """Test det curvature - generally unknown."""
    X = Variable(shape=(3, 3), name="X")
    d = det(X)
    # det of general matrix is unknown
    assert d.curvature == nvx.UNKNOWN
