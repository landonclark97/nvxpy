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
    from nvxpy.sets.discrete_set import DiscreteSet

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
    from nvxpy.sets.discrete_set import DiscreteSet

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
    from nvxpy.sets.discrete_set import DiscreteSet

    ds = DiscreteSet([1, 5, 10, 20])
    assert ds.nearest(3) == 1.0 or ds.nearest(3) == 5.0  # 3 is equidistant
    assert ds.nearest(7) == 5.0
    assert ds.nearest(16) == 20.0


def test_discrete_set_constraint():
    from nvxpy.sets.discrete_set import DiscreteSet

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
    prob = nvx.Problem(nvx.Minimize((x - 7) ** 2), [x ^ [1, 5, 10, 15]])

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 5.0).all()


def test_discrete_set_solve_float():
    """Test solving with discrete float constraint."""
    nvx.reset_variable_ids()
    y = Variable(name="y")

    # Minimize (y - 0.7)^2 subject to y in {0.1, 0.5, 1.0, 2.5}
    # Closest to 0.7 is either 0.5 (dist=0.2) or 1.0 (dist=0.3)
    # Optimal: y = 0.5
    prob = nvx.Problem(nvx.Minimize((y - 0.7) ** 2), [y ^ [0.1, 0.5, 1.0, 2.5]])

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert abs(y.value - 0.5) < 1e-5


def test_discrete_set_multiple_vars():
    """Test solving with multiple discrete variables."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")

    # Minimize x + y subject to x in {2, 4, 6} and y in {1, 3, 5}
    # Optimal: x=2, y=1, objective=3
    prob = nvx.Problem(nvx.Minimize(x + y), [x ^ [2, 4, 6], y ^ [1, 3, 5]])

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value == 2.0
    assert y.value == 1.0


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
        ],
    )

    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value + y.value >= 5 - 1e-5


def test_discrete_set_integer_var_with_floats_raises():
    """Test that integer variable with non-integer discrete set raises error."""
    import pytest

    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")

    with pytest.raises(ValueError, match="non-integer values"):
        x ^ [1.2, 2.2, 4.2, -2.2]


def test_discrete_set_continuous_var_with_floats_ok():
    """Test that continuous variable with non-integer discrete set is allowed."""
    nvx.reset_variable_ids()
    x = Variable(name="x")  # continuous

    # This should not raise
    cons = x ^ [1.2, 2.2, 4.2, -2.2]
    assert cons.op == "in"


# =============================================================================
# n-D DiscreteSet Tests (DiscretePoints)
# =============================================================================


def test_discrete_set_nd_initialization():
    """Test DiscreteSet with n-dimensional points."""
    from nvxpy.sets.discrete_set import DiscreteSet

    # 2D points
    ds = DiscreteSet([[1, 2], [3, 4], [5, 6]])
    assert ds.point_dim == 2
    assert len(ds) == 3
    assert ds.values == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))


def test_discrete_set_nd_membership():
    """Test n-D point membership checking."""
    from nvxpy.sets.discrete_set import DiscreteSet

    ds = DiscreteSet([[1, 2], [3, 4], [5, 6]])
    assert [1, 2] in ds
    assert [3, 4] in ds
    assert [5, 6] in ds
    assert [1, 3] not in ds
    assert [1.0000001, 2.0000001] in ds  # Within default tolerance


def test_discrete_set_nd_nearest():
    """Test finding nearest n-D point."""
    from nvxpy.sets.discrete_set import DiscreteSet

    ds = DiscreteSet([[0, 0], [10, 0], [0, 10]])
    # Point (3, 2) is closest to (0, 0)
    assert ds.nearest([3, 2]) == (0.0, 0.0)
    # Point (7, 1) is closest to (10, 0)
    assert ds.nearest([7, 1]) == (10.0, 0.0)


def test_discrete_set_nd_constraint():
    """Test n-D discrete constraint creation."""
    x = Variable(shape=(2,), name="x")
    cons = x ^ [[1, 2], [3, 4], [5, 6]]
    assert isinstance(cons, Constraint)
    assert cons.op == "in"


def test_discrete_set_nd_solve():
    """Test solving with n-D discrete constraint."""
    nvx.reset_variable_ids()
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    # Minimize distance to (2.3, 3.8)
    # Points: [1,3], [3,4], [4,1]
    # Distances: sqrt((2.3-1)^2 + (3.8-3)^2) = sqrt(1.69 + 0.64) = 1.53
    #            sqrt((2.3-3)^2 + (3.8-4)^2) = sqrt(0.49 + 0.04) = 0.73  <- closest
    #            sqrt((2.3-4)^2 + (3.8-1)^2) = sqrt(2.89 + 7.84) = 3.27
    prob = nvx.Problem(
        nvx.Minimize((x[0] - 2.3) ** 2 + (x[1] - 3.8) ** 2),
        [x ^ [[1, 3], [3, 4], [4, 1]]],
    )
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, [3.0, 4.0], atol=1e-3)


def test_discrete_set_nd_3d_points():
    """Test solving with 3D discrete points."""
    nvx.reset_variable_ids()
    x = Variable(shape=(3,), name="x")
    x.value = np.array([0.0, 0.0, 0.0])

    # 3D points
    points = [[0, 0, 0], [1, 1, 1], [2, 0, 1], [0, 2, 1]]
    # Target: (1.1, 0.9, 0.8) -> closest is [1, 1, 1]
    prob = nvx.Problem(
        nvx.Minimize((x[0] - 1.1) ** 2 + (x[1] - 0.9) ** 2 + (x[2] - 0.8) ** 2),
        [x ^ points],
    )
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, [1.0, 1.0, 1.0], atol=1e-3)


# =============================================================================
# DiscreteSet dimension mismatch tests (negative cases)
# =============================================================================


def test_discrete_set_dimension_mismatch_scalar_with_3d_points():
    """Test that scalar variable with 3D points raises error."""
    import pytest

    nvx.reset_variable_ids()
    x = Variable(name="x")  # scalar (size 1)

    # 3D points don't match scalar variable (2D would be interpreted as ranges)
    with pytest.raises(ValueError, match="does not match"):
        x ^ [[1, 2, 3], [4, 5, 6]]


def test_discrete_set_dimension_mismatch_nd_with_wrong_dim():
    """Test that n-D variable with wrong dimension points raises error."""
    import pytest

    nvx.reset_variable_ids()
    x = Variable(shape=(3,), name="x")  # 3D variable

    # 2D points don't match 3D variable
    with pytest.raises(ValueError, match="does not match"):
        x ^ [[1, 2], [3, 4]]


def test_discrete_set_dimension_mismatch_nd_with_scalars():
    """Test that n-D variable with scalar values raises error."""
    import pytest

    nvx.reset_variable_ids()
    x = Variable(shape=(2,), name="x")  # 2D variable

    # Scalar values don't match 2D variable
    with pytest.raises(ValueError, match="does not match"):
        x ^ [1, 2, 3]


def test_discrete_set_inconsistent_point_shapes():
    """Test that inconsistent point shapes raise error."""
    import pytest
    from nvxpy.sets.discrete_set import DiscreteSet

    # Points have different shapes
    with pytest.raises(ValueError, match="same shape"):
        DiscreteSet([[1, 2], [3, 4, 5]])


# =============================================================================
# DiscreteRanges Tests (with indicator variable reformulation)
# =============================================================================


def test_discrete_ranges_solve():
    """Test solving with DiscreteRanges constraint using indicator reformulation."""
    nvx.reset_variable_ids()
    x = Variable(name="x")
    x.value = 0.0

    # x must be in one of: [0, 2], [5, 7], [10, 12]
    # Minimize (x - 6)^2 -> optimal is x = 6 (in range [5, 7])
    prob = nvx.Problem(nvx.Minimize((x - 6) ** 2), [x ^ [[0, 2], [5, 7], [10, 12]]])
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 6.0, atol=1e-3).all()


def test_discrete_ranges_at_boundary():
    """Test DiscreteRanges where optimal is at range boundary."""
    nvx.reset_variable_ids()
    x = Variable(name="x")
    x.value = 0.0

    # x must be in one of: [0, 2], [10, 12]
    # Minimize (x - 5)^2 -> optimal is x = 2 (upper boundary of first range)
    prob = nvx.Problem(nvx.Minimize((x - 5) ** 2), [x ^ [[0, 2], [10, 12]]])
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    # Distance to 5 from 2 is 3, from 10 is 5, so x=2 is optimal
    assert np.isclose(x.value, 2.0, atol=1e-3).all()


def test_discrete_ranges_multiple_vars():
    """Test DiscreteRanges with multiple variables."""
    nvx.reset_variable_ids()
    x = Variable(name="x")
    y = Variable(name="y")
    x.value = 0.0
    y.value = 0.0

    # x in [0, 1] or [5, 6], y in [0, 2] or [8, 10]
    # Minimize x + y -> optimal is x=0, y=0
    prob = nvx.Problem(
        nvx.Minimize(x + y), [x ^ [[0, 1], [5, 6]], y ^ [[0, 2], [8, 10]]]
    )
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 0.0, atol=1e-3).all()
    assert np.isclose(y.value, 0.0, atol=1e-3).all()


def test_discrete_ranges_with_other_constraints():
    """Test DiscreteRanges combined with regular constraints."""
    nvx.reset_variable_ids()
    x = Variable(name="x")
    y = Variable(name="y")
    x.value = 0.0
    y.value = 0.0

    # x in [0, 2] or [5, 7], x + y >= 8, y >= 0
    # Minimize x + y -> need x + y >= 8
    # If x in [5, 7]: x=5, y=3 -> obj=8
    # If x in [0, 2]: x=2, y=6 -> obj=8
    # Both give same objective, but range [5,7] should work
    prob = nvx.Problem(nvx.Minimize(x + y), [x ^ [[0, 2], [5, 7]], x + y >= 8, y >= 0])
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value + y.value >= 8 - 1e-3


# =============================================================================
# Expression constraint tests
# =============================================================================


def test_discrete_set_on_expression():
    """Test DiscreteSet constraint on an expression (not just a variable)."""
    nvx.reset_variable_ids()
    x = Variable(name="x")
    y = Variable(name="y")
    x.value = 0.0
    y.value = 0.0

    # (x + y) must be in {1, 2, 3}
    # Minimize x^2 + y^2 subject to x + y in {1, 2, 3}
    # Optimal: x + y = 1 with x = y = 0.5 -> obj = 0.5
    prob = nvx.Problem(nvx.Minimize(x**2 + y**2), [(x + y) ^ [1, 2, 3]])
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value + y.value, 1.0, atol=1e-3).all()
    assert np.isclose(x.value, 0.5, atol=1e-3).all()
    assert np.isclose(y.value, 0.5, atol=1e-3).all()


def test_discrete_ranges_on_expression():
    """Test DiscreteRanges constraint on an expression."""
    nvx.reset_variable_ids()
    x = Variable(name="x")
    y = Variable(name="y")
    x.value = 0.0
    y.value = 0.0

    # (x + y) must be in [0, 1] or [5, 6]
    # Minimize (x + y - 3)^2 -> optimal is x + y at boundary closest to 3
    # From range [0,1]: closest is 1, distance = 4
    # From range [5,6]: closest is 5, distance = 4
    # Should pick one of these
    prob = nvx.Problem(nvx.Minimize((x + y - 3) ** 2), [(x + y) ^ [[0, 1], [5, 6]]])
    result = prob.solve(solver=nvx.BNB)
    assert result.status == nvx.SolverStatus.OPTIMAL
    sum_val = np.ravel(x.value)[0] + np.ravel(y.value)[0]
    assert np.isclose(sum_val, 1.0, atol=1e-3) or np.isclose(sum_val, 5.0, atol=1e-3)


# =============================================================================
# Projection Constraint Tests
# =============================================================================


def test_projection_constraint_residual_is_norm():
    """Test that projection constraint residual is ||x - proj(x)||."""
    from nvxpy.problem import Problem

    nvx.reset_variable_ids()
    x = Variable(shape=(3, 3), name="x")
    x.value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

    # Create projection constraint: x <- proj(x)
    so = SO(3)
    cons = so.constrain(x)
    assert cons.op == "<-"

    # Build the problem to get constraint functions
    prob = Problem(Minimize(nvx.sum(x)), [cons])

    # Find the projection constraint
    proj_cons = [c for c in prob._constraint_fns if c.op == "<-"]
    assert len(proj_cons) == 1

    # Evaluate: should return ||x - proj(x)||
    flat_x = np.ravel(x.value)
    residual = proj_cons[0].fun(flat_x)

    # Residual should be a single non-negative scalar (the norm)
    assert residual.shape == (1,)
    assert residual[0] >= 0


def test_eval_projection_constraint_helper():
    """Test eval_projection_constraint applies p_tol correctly."""
    from nvxpy.problem import Problem
    from nvxpy.solvers.base import eval_projection_constraint

    nvx.reset_variable_ids()
    x = Variable(shape=(3, 3), name="x")
    # Start with non-orthogonal matrix
    x.value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

    so = SO(3)
    cons = so.constrain(x)
    prob = Problem(Minimize(nvx.sum(x)), [cons])

    proj_cons = [c for c in prob._constraint_fns if c.op == "<-"][0]
    flat_x = np.ravel(x.value)

    # Get the norm
    norm = proj_cons.fun(flat_x)[0]

    # Test with different p_tol values
    # With small p_tol, residual should be negative (constraint violated)
    small_tol = 1e-10
    residual_small = eval_projection_constraint(proj_cons, flat_x, small_tol)
    assert residual_small[0] < 0  # Violated

    # With large p_tol, residual should be positive (constraint satisfied)
    large_tol = 1000.0
    residual_large = eval_projection_constraint(proj_cons, flat_x, large_tol)
    assert residual_large[0] > 0  # Satisfied

    # Exact tolerance should give zero residual
    exact_tol = norm
    residual_exact = eval_projection_constraint(proj_cons, flat_x, exact_tol)
    assert np.isclose(residual_exact[0], 0, atol=1e-10)


def test_projection_constraint_with_custom_p_tol():
    """Test that p_tol option affects projection constraint evaluation."""
    nvx.reset_variable_ids()
    x = Variable(shape=(3, 3), name="x")
    # Start close to orthogonal
    x.value = np.eye(3) + 0.01 * np.random.randn(3, 3)

    so = SO(3)
    prob = nvx.Problem(nvx.Minimize(nvx.norm(x - np.eye(3), ord="fro")), [x ^ so])

    # With default p_tol, should converge
    result = prob.solve(solver_options={"p_tol": 1e-6})
    assert result.status == nvx.SolverStatus.OPTIMAL

    # Final x should be close to orthogonal
    assert np.allclose(x.value.T @ x.value, np.eye(3), atol=1e-4)
