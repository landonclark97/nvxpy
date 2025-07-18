import autograd.numpy as np
import nvxpy as nvx
from nvxpy.sets.special_orthogonal import SO
from nvxpy.sets.perspective_cone import PerspectiveCone
from nvxpy.constraint import Constraint
from nvxpy.atoms.polar import PolarDecomposition
from nvxpy.problem import Problem, Minimize
from nvxpy.variable import Variable


def test_so_initialization():
    n = 3
    so = SO(n)
    assert so.n == n
    assert so.name == f"SO({n})"


def test_so_constrain():
    n = 3
    so = SO(n)
    var = Variable(shape=(n, n), name="some_variable")
    constraints = so.constrain(var)
    assert constraints.op == "<-"
    assert constraints.left == nvx.norm(var - PolarDecomposition(var), ord="fro")
    assert constraints.right == 1e-8


def test_so_constraint_w_problem():
    n = 3
    so_n = SO(n)
    var = Variable(shape=(n, n), name="some_variable")
    var.value = np.random.uniform(-2, 2, (n, n))
    obj = nvx.norm(var - np.eye(n), ord="fro")
    problem = Problem(Minimize(obj), [var ^ so_n])
    problem.solve()

    assert np.allclose(var.value.T @ var.value, np.eye(n), atol=1e-5)
    assert np.isclose(np.linalg.det(var.value), 1)


def test_perspective_cone_initialization():
    func = nvx.norm
    expr = Variable(shape=(3,))
    expr.value = np.array([1, 2, 3])
    p = Variable()
    p.value = 1.0
    pc = PerspectiveCone(func, expr, p)
    assert pc.func == func
    assert np.array_equal(pc.expr, expr)
    assert pc.p == p


def test_perspective_cone_constrain():
    func = nvx.norm
    expr = Variable(shape=(3,))
    expr.value = np.array([1, 2, 3])
    p = Variable()
    p.value = 1.0
    pc = PerspectiveCone(func, expr, p)
    var = Variable(shape=(3,), name="some_variable")
    constraint = pc.constrain(var)
    assert isinstance(constraint, Constraint)
    expected_expr = p * func(expr / (p + 1e-8))
    assert constraint.op == "=="
    assert constraint.right == expected_expr
