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
from nvxpy.atoms.log import log
from nvxpy.atoms.exp import exp
from nvxpy.atoms.sqrt import sqrt
from nvxpy.atoms.minimum import minimum
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


def test_log():
    """Test log atom."""
    x = np.array([1.0, np.e, np.e**2])
    log_expr = log(x)
    assert np.allclose(log_expr(x), np.array([0.0, 1.0, 2.0]))


def test_log_curvature():
    """Test log curvature analysis."""
    x = Variable(shape=(3,), name="x")
    log_expr = log(x)
    # log of affine is concave
    assert log_expr.curvature == nvx.CONCAVE

    # log of concave is concave (composition rule)
    neg_sq = -(x ** 2)  # concave
    l_concave = log(neg_sq)
    assert l_concave.curvature == nvx.CONCAVE

    # log of constant is constant
    const = np.array([1.0, 2.0, 3.0])
    l_const = log(const)
    assert l_const.curvature == nvx.CONSTANT

    # log of convex is unknown
    sq = x ** 2  # convex
    l_convex = log(sq)
    assert l_convex.curvature == nvx.UNKNOWN


def test_exp():
    """Test exp atom."""
    x = np.array([0.0, 1.0, 2.0])
    e = exp(x)
    assert np.allclose(e(x), np.array([1.0, np.e, np.e**2]))


def test_exp_curvature():
    """Test exp curvature analysis."""
    x = Variable(shape=(3,), name="x")
    e = exp(x)
    # exp of affine is convex
    assert e.curvature == nvx.CONVEX

    # exp of convex is convex (composition rule)
    sq = x ** 2  # convex
    e_convex = exp(sq)
    assert e_convex.curvature == nvx.CONVEX

    # exp of constant is constant
    const = np.array([0.0, 1.0, 2.0])
    e_const = exp(const)
    assert e_const.curvature == nvx.CONSTANT

    # exp of concave is unknown
    neg_sq = -(x ** 2)  # concave
    e_concave = exp(neg_sq)
    assert e_concave.curvature == nvx.UNKNOWN


def test_sqrt():
    """Test sqrt atom."""
    x = np.array([1.0, 4.0, 9.0])
    s = sqrt(x)
    assert np.allclose(s(x), np.array([1.0, 2.0, 3.0]))


def test_sqrt_curvature():
    """Test sqrt curvature analysis."""
    x = Variable(shape=(3,), name="x")
    s = sqrt(x)
    # sqrt of affine is concave
    assert s.curvature == nvx.CONCAVE

    # sqrt of concave is concave (composition rule)
    neg_sq = -(x ** 2)  # concave
    s_concave = sqrt(neg_sq)
    assert s_concave.curvature == nvx.CONCAVE

    # sqrt of constant is constant
    const = np.array([1.0, 4.0, 9.0])
    s_const = sqrt(const)
    assert s_const.curvature == nvx.CONSTANT

    # sqrt of convex is unknown
    sq = x ** 2  # convex
    s_convex = sqrt(sq)
    assert s_convex.curvature == nvx.UNKNOWN


def test_minimum():
    """Test minimum atom."""
    x = np.array([1, 3, 5])
    y = np.array([2, 2, 6])
    m = minimum(x, y)
    assert np.array_equal(m(x, y), np.array([1, 2, 5]))


def test_minimum_curvature():
    """Test minimum curvature analysis."""
    x = Variable(shape=(3,), name="x")
    y = Variable(shape=(3,), name="y")
    m = minimum(x, y)
    # minimum of two affine is concave
    assert m.curvature == nvx.CONCAVE

    # minimum of concave and affine is concave
    neg_sq = -(x ** 2)  # concave
    m_mixed = minimum(neg_sq, y)
    assert m_mixed.curvature == nvx.CONCAVE

    # minimum of constant and constant is constant
    const1 = np.array([1, 2, 3])
    const2 = np.array([2, 1, 4])
    m_const = minimum(const1, const2)
    assert m_const.curvature == nvx.CONSTANT

    # minimum involving convex is unknown
    sq = x ** 2  # convex
    m_convex = minimum(sq, y)
    assert m_convex.curvature == nvx.UNKNOWN
