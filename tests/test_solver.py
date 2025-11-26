import autograd.numpy as np
import nvxpy as nvx
from nvxpy.problem import Problem, Minimize
from nvxpy.variable import Variable


def test_convex_problem():
    # Define variables
    x = Variable(shape=(2,), name="x")

    # Define objective: Minimize (x1 - 1)^2 + (x2 - 2)^2
    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

    # Define constraints: x1 + x2 = 3
    constraints = [x[0] + x[1] == 3]

    # Create and solve problem
    problem = Problem(objective, constraints)
    problem.solve()

    # Assert solution
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-4)


def test_non_convex_obj():
    # Define variables
    x = Variable(shape=(2,), name="x")

    # Define objective: Minimize sin(x1) + cos(x2)
    def cos_sin_obj(x):
        return np.sin(x[0]) + np.cos(x[1])

    obj_func = nvx.Function(cos_sin_obj, jac="autograd")

    objective = Minimize(obj_func(x))

    # Define constraints: x1^2 + x2^2 <= 1
    constraints = [x[0] ** 2 + x[1] ** 2 <= 1]

    # Create and solve problem
    problem = Problem(objective, constraints)
    problem.solve()

    # Assert solution is within the unit circle
    assert x.value[0] ** 2 + x.value[1] ** 2 <= 1


def test_non_convex_constraints():
    # Define variables
    x = Variable(shape=(2,), name="x")
    x.value = np.array([2.0, 0.0])

    x_d = np.array([-2.0, 0.0])

    objective = Minimize(nvx.norm(x - x_d))

    # Define constraints: x1^2 + x2^2 <= 1
    constraints = [nvx.norm(x) >= 1]

    # Create and solve problem
    problem = Problem(objective, constraints)
    problem.solve()

    # Assert solution is within the unit circle
    assert x.value[0] ** 2 + x.value[1] ** 2 >= 1
    assert not np.allclose(x.value, x_d, atol=1e-4)

    x.value = np.array([0.0, 2.0])
    problem.solve()
    assert x.value[0] ** 2 + x.value[1] ** 2 >= 1
    assert np.allclose(x.value, x_d, atol=1e-4)

    x.value = np.array([0.0, -2.0])
    problem.solve()
    assert x.value[0] ** 2 + x.value[1] ** 2 >= 1
    assert np.allclose(x.value, x_d, atol=1e-4)


def test_maximize_and_presolve():
    X = Variable((4,4))
    X.value = np.arange(16).reshape(4,4)

    obj = nvx.norm(X - np.eye(4), ord="fro")
    cons = [
        nvx.norm(X, ord="fro") <= 3,
        X >> 0,
    ]
    problem = Problem(nvx.Maximize(obj), cons)
    problem.solve(presolve=True)

    assert problem.status == nvx.SolverStatus.OPTIMAL


def test_ipopt_solver():
    """Test IPOPT solver via cyipopt."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    # Minimize (x1 - 1)^2 + (x2 - 2)^2 subject to x1 + x2 >= 1
    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 1, x[0] >= 0, x[1] >= 0]

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.IPOPT)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


def test_ipopt_unconstrained():
    """Test IPOPT solver on unconstrained problem."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    x = Variable(shape=(2,), name="x")
    x.value = np.array([5.0, 5.0])

    # Simple unconstrained quadratic
    objective = Minimize((x[0] - 3) ** 2 + (x[1] + 1) ** 2)

    problem = Problem(objective, [])
    result = problem.solve(solver=nvx.IPOPT)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([3.0, -1.0]), atol=1e-3)
