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


def test_ipopt_equality_constraint():
    """Test IPOPT with equality constraints."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    # Minimize x1^2 + x2^2 subject to x1 + x2 = 1
    objective = Minimize(x[0] ** 2 + x[1] ** 2)
    constraints = [x[0] + x[1] == 1]

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.IPOPT)

    assert result.status == nvx.SolverStatus.OPTIMAL
    # Solution should be x1 = x2 = 0.5
    assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-3)


def test_ipopt_with_compile():
    """Test IPOPT with expression compilation enabled."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 0]

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.IPOPT, compile=True)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


def test_slsqp_solver():
    """Test SLSQP solver explicitly."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 0]

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.SLSQP)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


def test_cobyla_solver():
    """Test COBYLA solver."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 1]

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.COBYLA)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-2)


def test_bfgs_solver():
    """Test BFGS solver (unconstrained)."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    # BFGS is unconstrained
    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

    problem = Problem(objective, [])
    result = problem.solve(solver=nvx.BFGS)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


def test_solver_with_options():
    """Test solver with custom options."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

    problem = Problem(objective, [])
    result = problem.solve(
        solver=nvx.SLSQP, solver_options={"maxiter": 100, "ftol": 1e-10}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_solver_result_stats():
    """Test that solver stats are populated."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

    problem = Problem(objective, [])
    result = problem.solve()

    # Check stats exist
    assert result.stats is not None
    assert result.stats.solver_name is not None
    assert result.stats.solve_time is not None or result.stats.solve_time >= 0


def test_problem_with_compile():
    """Test problem with compilation enabled."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 0]

    problem = Problem(objective, constraints)
    result = problem.solve(compile=True)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


def test_ipopt_status_interpretation():
    """Test IPOPT status code interpretation."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    from nvxpy.solvers.ipopt_backend import IpoptBackend

    backend = IpoptBackend()

    # Test various status codes
    assert backend._interpret_status({"status": 0}) == nvx.SolverStatus.OPTIMAL
    assert backend._interpret_status({"status": 1}) == nvx.SolverStatus.SUBOPTIMAL
    assert backend._interpret_status({"status": 2}) == nvx.SolverStatus.INFEASIBLE
    assert backend._interpret_status({"status": 3}) == nvx.SolverStatus.NUMERICAL_ERROR
    assert backend._interpret_status({"status": 4}) == nvx.SolverStatus.UNBOUNDED
    assert backend._interpret_status({"status": 5}) == nvx.SolverStatus.ERROR
    assert backend._interpret_status({"status": 6}) == nvx.SolverStatus.OPTIMAL
    assert backend._interpret_status({"status": -1}) == nvx.SolverStatus.MAX_ITERATIONS
    assert backend._interpret_status({"status": -2}) == nvx.SolverStatus.NUMERICAL_ERROR
    assert backend._interpret_status({"status": -3}) == nvx.SolverStatus.NUMERICAL_ERROR
    assert backend._interpret_status({"status": -4}) == nvx.SolverStatus.MAX_ITERATIONS
    assert backend._interpret_status({"status": -5}) == nvx.SolverStatus.MAX_ITERATIONS
    assert backend._interpret_status({"status": -10}) == nvx.SolverStatus.ERROR
    assert backend._interpret_status({"status": -11}) == nvx.SolverStatus.ERROR
    assert backend._interpret_status({"status": -12}) == nvx.SolverStatus.ERROR
    assert backend._interpret_status({"status": -13}) == nvx.SolverStatus.NUMERICAL_ERROR
    assert backend._interpret_status({"status": -999}) == nvx.SolverStatus.UNKNOWN


def test_nelder_mead_solver():
    """Test Nelder-Mead solver (derivative-free)."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    # Nelder-Mead is unconstrained
    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

    problem = Problem(objective, [])
    result = problem.solve(solver=nvx.NELDER_MEAD)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-2)


def test_trust_constr_solver():
    """Test trust-constr solver."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 1]

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.TRUST_CONSTR)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-2)
