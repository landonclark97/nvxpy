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


# =============================================================================
# DiscreteSet Tests
# =============================================================================

def test_discrete_set_initialization():
    from nvxpy.sets.integer_set import DiscreteSet

    # Integer values
    ds = DiscreteSet([1, 5, 10, 3])
    assert ds.values == (1.0, 3.0, 5.0, 10.0)  # Sorted
    assert len(ds) == 4

    # Float values
    ds_float = DiscreteSet([0.1, 0.5, 1.0, 2.5])
    assert ds_float.values == (0.1, 0.5, 1.0, 2.5)

    # Duplicates are removed
    ds_dup = DiscreteSet([1, 1, 2, 2, 3])
    assert ds_dup.values == (1.0, 2.0, 3.0)


def test_discrete_set_membership():
    from nvxpy.sets.integer_set import DiscreteSet

    ds = DiscreteSet([1, 5, 10])
    assert 1 in ds
    assert 5 in ds
    assert 10 in ds
    assert 2 not in ds
    assert 1.0000001 in ds  # Within default tolerance (1e-6)

    # Custom tolerance
    ds_tol = DiscreteSet([1, 5, 10], tolerance=1e-3)
    assert 1.0001 in ds_tol  # Within custom tolerance


def test_discrete_set_nearest():
    from nvxpy.sets.integer_set import DiscreteSet

    ds = DiscreteSet([1, 5, 10, 20])
    assert ds.nearest(3) == 1.0 or ds.nearest(3) == 5.0  # 3 is equidistant
    assert ds.nearest(7) == 5.0
    assert ds.nearest(16) == 20.0


def test_discrete_set_constraint():
    from nvxpy.sets.integer_set import DiscreteSet

    var = Variable(integer=True, name="x")
    ds = DiscreteSet([1, 5, 10])
    constraint = var ^ ds
    assert isinstance(constraint, Constraint)
    assert constraint.op == "in"
    assert constraint.left == var
    assert constraint.right == ds


def test_discrete_set_list_shorthand():
    """Test that x ^ [list] syntax works."""
    var = Variable(integer=True, name="x")
    constraint = var ^ [1, 5, 10]
    assert isinstance(constraint, Constraint)
    assert constraint.op == "in"
    assert constraint.right.values == (1.0, 5.0, 10.0)


def test_discrete_constraint_curvature():
    """Discrete set membership is non-convex."""
    from nvxpy.constants import Curvature

    var = Variable(integer=True, name="x")
    constraint = var ^ [1, 5, 10]
    assert constraint.curvature == Curvature.UNKNOWN


def test_discrete_set_solve_integer():
    """Test solving with discrete integer constraint."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")

    # Minimize (x - 7)^2 subject to x in {1, 5, 10, 15}
    # Optimal: x = 5 (closest to 7 from below) or x = 10 (from above)
    # Since 7-5=2 < 10-7=3, optimal is x=5
    prob = nvx.Problem(
        nvx.Minimize((x - 7)**2),
        [x ^ [1, 5, 10, 15]]
    )

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value.item() == 5.0


def test_discrete_set_solve_float():
    """Test solving with discrete float constraint."""
    nvx.reset_variable_ids()
    y = Variable(name="y")

    # Minimize (y - 0.7)^2 subject to y in {0.1, 0.5, 1.0, 2.5}
    # Closest to 0.7 is either 0.5 (dist=0.2) or 1.0 (dist=0.3)
    # Optimal: y = 0.5
    prob = nvx.Problem(
        nvx.Minimize((y - 0.7)**2),
        [y ^ [0.1, 0.5, 1.0, 2.5]]
    )

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert abs(y.value.item() - 0.5) < 1e-5


def test_discrete_set_multiple_vars():
    """Test solving with multiple discrete variables."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")

    # Minimize x + y subject to x in {2, 4, 6} and y in {1, 3, 5}
    # Optimal: x=2, y=1, objective=3
    prob = nvx.Problem(
        nvx.Minimize(x + y),
        [x ^ [2, 4, 6], y ^ [1, 3, 5]]
    )

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value.item() == 2.0
    assert y.value.item() == 1.0


def test_discrete_set_with_other_constraints():
    """Test discrete constraints combined with regular constraints."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(name="y")

    # Minimize x + y subject to:
    #   x in {1, 3, 5, 7}
    #   x + y >= 5
    #   y >= 0
    # Optimal: x=5, y=0 (objective=5) or x=3, y=2 (objective=5), etc.
    prob = nvx.Problem(
        nvx.Minimize(x + y),
        [
            x ^ [1, 3, 5, 7],
            x + y >= 5,
            y >= 0,
        ]
    )

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value.item() + y.value.item() >= 5 - 1e-5
