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
    X = Variable((4, 4))
    X.value = np.arange(16).reshape(4, 4)

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

    problem = Problem(objective, constraints, compile=True)
    result = problem.solve(solver=nvx.IPOPT)

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
    assert result.stats.solve_time is not None and result.stats.solve_time >= 0


def test_problem_with_compile():
    """Test problem with compilation enabled."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 0]

    problem = Problem(objective, constraints, compile=True)
    result = problem.solve()

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
    assert (
        backend._interpret_status({"status": -13}) == nvx.SolverStatus.NUMERICAL_ERROR
    )
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


def test_default_solver_unconstrained():
    """Test default solver selection for unconstrained problem (L-BFGS-B)."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

    problem = Problem(objective, [])
    result = problem.solve()  # No solver specified

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert result.stats.solver_name == "L-BFGS-B"
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


def test_default_solver_constrained():
    """Test default solver selection for constrained problem (SLSQP)."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    constraints = [x[0] + x[1] >= 1]

    problem = Problem(objective, constraints)
    result = problem.solve()  # No solver specified

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert result.stats.solver_name == "SLSQP"
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


def test_default_solver_integer():
    """Test default solver selection for integer problem (BNB)."""
    x = Variable(shape=(2,), name="x", integer=True)
    x.value = np.array([0.0, 0.0])

    objective = Minimize((x[0] - 1.2) ** 2 + (x[1] - 2.7) ** 2)

    problem = Problem(objective, [])
    result = problem.solve()  # No solver specified

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert "B&B" in result.stats.solver_name
    # Should snap to nearest integers: 1 and 3
    assert np.allclose(x.value, np.array([1.0, 3.0]), atol=1e-3)


def test_default_solver_discrete_set():
    """Test default solver selection for DiscreteSet constraint (BNB)."""
    # Scalar variable with scalar discrete values
    x = Variable(shape=(1,), name="x")
    x.value = np.array([0.0])

    objective = Minimize((x - 1.7) ** 2)
    # DiscreteSet constraint without integer=True on variable
    constraints = [x ^ [0, 1, 2, 3, 4, 5]]

    problem = Problem(objective, constraints)
    result = problem.solve()  # No solver specified

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert "B&B" in result.stats.solver_name
    # Should snap to nearest discrete value: 2
    assert np.isclose(x.value, 2.0).all()


def test_discrete_set_nd_points():
    """Test DiscreteSet with n-dimensional points."""
    x = Variable(shape=(2,), name="x")
    x.value = np.array([0.0, 0.0])

    # x can be one of these 2D points: [0,0], [1,3], [3,1], [2,2]
    objective = Minimize((x[0] - 1.2) ** 2 + (x[1] - 2.7) ** 2)
    constraints = [x ^ [[0, 0], [1, 3], [3, 1], [2, 2]]]

    problem = Problem(objective, constraints)
    result = problem.solve()

    assert result.status == nvx.SolverStatus.OPTIMAL
    # Closest point to (1.2, 2.7) is [1, 3] with distance sqrt(0.04 + 0.09) = 0.36
    # vs [2,2] with distance sqrt(0.64 + 0.49) = 1.06
    assert np.allclose(x.value, np.array([1.0, 3.0]), atol=1e-3)


def test_ipopt_psd_constraint():
    """Test IPOPT with positive semidefinite constraint."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    X = Variable(shape=(2, 2), name="X")
    X.value = np.eye(2)

    # Minimize Frobenius norm from target, subject to PSD constraint
    target = np.array([[2.0, 1.0], [1.0, 2.0]])
    objective = Minimize(nvx.norm(X - target, ord="fro"))
    constraints = [X >> 0]  # X must be positive semidefinite

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.IPOPT)

    assert result.status == nvx.SolverStatus.OPTIMAL
    # Check that eigenvalues are non-negative (PSD)
    eigenvalues = np.linalg.eigvalsh(X.value)
    assert np.all(eigenvalues >= -1e-6)


def test_ipopt_nsd_constraint():
    """Test IPOPT with negative semidefinite constraint."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    X = Variable(shape=(2, 2), name="X")
    X.value = -np.eye(2)

    # Minimize Frobenius norm from target, subject to NSD constraint
    target = np.array([[-2.0, -0.5], [-0.5, -2.0]])
    objective = Minimize(nvx.norm(X - target, ord="fro"))
    constraints = [X << 0]  # X must be negative semidefinite

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.IPOPT)

    assert result.status == nvx.SolverStatus.OPTIMAL
    # Check that eigenvalues are non-positive (NSD)
    eigenvalues = np.linalg.eigvalsh(X.value)
    assert np.all(eigenvalues <= 1e-6)


def test_ipopt_projection_constraint():
    """Test IPOPT with projection constraint (SO(n) special orthogonal)."""
    pytest = __import__("pytest")

    try:
        import cyipopt  # noqa: F401
    except ImportError:
        pytest.skip("cyipopt not installed")

    from nvxpy.sets.special_orthogonal import SO

    R = Variable(shape=(2, 2), name="R")
    # Start with something close to a rotation matrix
    R.value = np.array([[0.9, -0.1], [0.1, 0.9]])

    # Target rotation (45 degrees)
    theta = np.pi / 4
    target = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    objective = Minimize(nvx.norm(R - target, ord="fro"))
    constraints = [R ^ SO(2)]  # R must be in SO(2)

    problem = Problem(objective, constraints)
    result = problem.solve(solver=nvx.IPOPT)

    assert result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]
    # Check that R is close to orthogonal (R^T @ R ≈ I)
    RtR = R.value.T @ R.value
    assert np.allclose(RtR, np.eye(2), atol=1e-2)
    # Check determinant is close to 1 (not -1, so it's a rotation, not reflection)
    assert np.abs(np.linalg.det(R.value) - 1.0) < 0.1


# =============================================================================
# Tests for new scipy solvers
# =============================================================================


class TestGradientFreeSolvers:
    """Tests for gradient-free scipy solvers."""

    def test_powell_solver(self):
        """Test Powell solver (derivative-free)."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.POWELL)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_cobyqa_solver(self):
        """Test COBYQA solver (constrained derivative-free)."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
        constraints = [x[0] + x[1] >= 1]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.COBYQA)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-2)


class TestGradientBasedSolvers:
    """Tests for gradient-based scipy solvers."""

    def test_cg_solver(self):
        """Test CG (conjugate gradient) solver."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.CG)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_tnc_solver(self):
        """Test TNC solver (unconstrained)."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.TNC)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_lbfgsb_solver(self):
        """Test L-BFGS-B solver."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.LBFGSB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


class TestHessianBasedSolvers:
    """Tests for Hessian-based scipy solvers."""

    def test_newton_cg_solver(self):
        """Test Newton-CG solver."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.NEWTON_CG)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_dogleg_solver(self):
        """Test dogleg solver with default trust region parameters."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.DOGLEG)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_dogleg_solver_custom_trust_radius(self):
        """Test dogleg solver with custom trust region parameters."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(
            solver=nvx.DOGLEG,
            solver_options={"initial_trust_radius": 0.5, "max_trust_radius": 2.0},
        )

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_trust_ncg_solver(self):
        """Test trust-ncg solver."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.TRUST_NCG)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_trust_krylov_solver(self):
        """Test trust-krylov solver."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.TRUST_KRYLOV)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_trust_exact_solver(self):
        """Test trust-exact solver."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.TRUST_EXACT)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


class TestHessianSolversRosenbrock:
    """Test Hessian-based solvers on Rosenbrock function (harder problem)."""

    def test_newton_cg_rosenbrock(self):
        """Test Newton-CG on Rosenbrock."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.NEWTON_CG)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-2)

    def test_trust_exact_rosenbrock(self):
        """Test trust-exact on Rosenbrock."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.TRUST_EXACT)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-2)

    def test_cg_rosenbrock(self):
        """Test CG on Rosenbrock."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.CG)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-2)
