"""Tests for KNITRO solver backend.

These tests require a valid KNITRO license to run.
"""

import autograd.numpy as np
import pytest

import nvxpy as nvx
from nvxpy.problem import Problem, Minimize
from nvxpy.variable import Variable


# Check if KNITRO is available and licensed
def knitro_available():
    try:
        import knitro

        # Try to create a context to verify license
        kc = knitro.KN_new()
        knitro.KN_free(kc)
        return True
    except ImportError:
        return False
    except RuntimeError:
        # License error
        return False


# Skip all tests if KNITRO is not installed or not licensed
pytestmark = pytest.mark.skipif(
    not knitro_available(), reason="KNITRO not installed or not licensed"
)


class TestKnitroBasic:
    """Basic KNITRO solver tests."""

    def test_knitro_unconstrained(self):
        """Test KNITRO solver on unconstrained problem."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([5.0, 5.0])

        objective = Minimize((x[0] - 3) ** 2 + (x[1] + 1) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([3.0, -1.0]), atol=1e-3)

    def test_knitro_inequality_constraint(self):
        """Test KNITRO with inequality constraints."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
        constraints = [x[0] + x[1] >= 1, x[0] >= 0, x[1] >= 0]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)

    def test_knitro_equality_constraint(self):
        """Test KNITRO with equality constraints."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        # Minimize x1^2 + x2^2 subject to x1 + x2 = 1
        objective = Minimize(x[0] ** 2 + x[1] ** 2)
        constraints = [x[0] + x[1] == 1]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([0.5, 0.5]), atol=1e-3)

    def test_knitro_mixed_constraints(self):
        """Test KNITRO with both equality and inequality constraints."""
        x = Variable(shape=(3,), name="x")
        x.value = np.array([1.0, 1.0, 1.0])

        # Minimize sum of squares
        objective = Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        constraints = [
            x[0] + x[1] + x[2] == 3,  # equality
            x[0] >= 0.5,  # inequality
            x[1] >= 0.5,  # inequality
        ]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.isclose(x.value.sum(), 3.0, atol=1e-3)
        assert x.value[0] >= 0.5 - 1e-3
        assert x.value[1] >= 0.5 - 1e-3


class TestKnitroNonlinear:
    """Test KNITRO with nonlinear problems."""

    def test_knitro_nonlinear_objective(self):
        """Test KNITRO with nonlinear objective using nvx.Function."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.5, 0.5])

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        obj_func = nvx.Function(rosenbrock, jac="autograd")
        objective = Minimize(obj_func(x))

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-2)

    def test_knitro_nonlinear_constraint(self):
        """Test KNITRO with nonlinear constraints."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.5, 0.5])

        objective = Minimize(x[0] + x[1])
        # Nonlinear constraint: x1^2 + x2^2 <= 1 (inside unit circle)
        constraints = [nvx.norm(x) <= 1]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        # Solution should be on the boundary
        assert np.isclose(np.linalg.norm(x.value), 1.0, atol=1e-2)


class TestKnitroInteger:
    """Test KNITRO's native MINLP support."""

    def test_knitro_integer_variables(self):
        """Test KNITRO with integer variables."""
        x = Variable(shape=(2,), name="x", integer=True)
        x.value = np.array([0.0, 0.0])

        # Solution should snap to integer values
        objective = Minimize((x[0] - 1.3) ** 2 + (x[1] - 2.7) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        # Should snap to [1, 3]
        assert np.allclose(x.value, np.array([1.0, 3.0]), atol=1e-3)

    def test_knitro_binary_variables(self):
        """Test KNITRO with binary variables."""
        x = Variable(shape=(3,), name="x", binary=True)
        x.value = np.array([0.0, 0.0, 0.0])

        # Knapsack-style problem
        values = np.array([3.0, 4.0, 2.0])
        weights = np.array([2.0, 3.0, 1.0])
        capacity = 4.0

        objective = Minimize(-nvx.sum(values * x))  # Maximize value
        constraints = [nvx.sum(weights * x) <= capacity]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        # All values should be 0 or 1
        assert np.all((x.value >= -1e-3) & (x.value <= 1 + 1e-3))
        # Should satisfy capacity constraint
        assert np.dot(weights, x.value) <= capacity + 1e-3

    def test_knitro_minlp(self):
        """Test KNITRO on mixed-integer nonlinear problem."""
        x_cont = Variable(shape=(1,), name="x_cont")
        x_int = Variable(shape=(1,), name="x_int", integer=True)

        x_cont.value = np.array([0.5])
        x_int.value = np.array([1.0])

        # Nonlinear objective with mixed variables
        objective = Minimize((x_cont[0] - 1.5) ** 2 + (x_int[0] - 2.3) ** 2)
        constraints = [x_cont[0] + x_int[0] >= 2]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        # x_int should be integer
        assert np.isclose(x_int.value[0] % 1, 0, atol=1e-3) or np.isclose(
            x_int.value[0] % 1, 1, atol=1e-3
        )


class TestKnitroOptions:
    """Test KNITRO solver options."""

    def test_knitro_custom_tolerances(self):
        """Test KNITRO with custom tolerances."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(
            solver=nvx.KNITRO, solver_options={"feastol": 1e-10, "opttol": 1e-10}
        )

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-5)

    def test_knitro_maxiter(self):
        """Test KNITRO with max iterations."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        result = problem.solve(solver=nvx.KNITRO, solver_options={"maxiter": 1000})

        assert result.status == nvx.SolverStatus.OPTIMAL

    def test_knitro_outlev(self):
        """Test KNITRO with output level control."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

        problem = Problem(objective, [])
        # outlev=0 should suppress output
        result = problem.solve(solver=nvx.KNITRO, solver_options={"outlev": 0})

        assert result.status == nvx.SolverStatus.OPTIMAL


class TestKnitroWithCompile:
    """Test KNITRO with expression compilation."""

    def test_knitro_with_compile(self):
        """Test KNITRO with expression compilation enabled."""
        x = Variable(shape=(2,), name="x")
        x.value = np.array([0.0, 0.0])

        objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
        constraints = [x[0] + x[1] >= 0]

        problem = Problem(objective, constraints, compile=True)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-3)


class TestKnitroProjection:
    """Test KNITRO with projection constraints."""

    def test_knitro_psd_constraint(self):
        """Test KNITRO with positive semidefinite constraint."""
        X = Variable(shape=(2, 2), name="X")
        X.value = np.eye(2)

        target = np.array([[2.0, 1.0], [1.0, 2.0]])
        objective = Minimize(nvx.norm(X - target, ord="fro"))
        constraints = [X >> 0]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status == nvx.SolverStatus.OPTIMAL
        eigenvalues = np.linalg.eigvalsh(X.value)
        assert np.all(eigenvalues >= -1e-6)

    def test_knitro_so2_projection(self):
        """Test KNITRO with SO(2) projection constraint."""
        from nvxpy.sets.special_orthogonal import SO

        R = Variable(shape=(2, 2), name="R")
        R.value = np.array([[0.9, -0.1], [0.1, 0.9]])

        theta = np.pi / 4
        target = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        objective = Minimize(nvx.norm(R - target, ord="fro"))
        constraints = [R ^ SO(2)]

        problem = Problem(objective, constraints)
        result = problem.solve(solver=nvx.KNITRO)

        assert result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]
        RtR = R.value.T @ R.value
        assert np.allclose(RtR, np.eye(2), atol=1e-2)
        assert np.abs(np.linalg.det(R.value) - 1.0) < 0.1


class TestKnitroStatusInterpretation:
    """Test KNITRO status code interpretation."""

    def test_status_interpretation(self):
        """Test KNITRO status code interpretation."""
        from nvxpy.solvers.knitro_backend import KnitroBackend

        backend = KnitroBackend()

        # Optimal
        assert backend._interpret_status(0) == nvx.SolverStatus.OPTIMAL
        # Near optimal / locally optimal
        assert backend._interpret_status(-100) == nvx.SolverStatus.OPTIMAL
        assert backend._interpret_status(-101) == nvx.SolverStatus.OPTIMAL
        assert backend._interpret_status(-102) == nvx.SolverStatus.OPTIMAL
        # Feasible point found
        assert backend._interpret_status(-103) == nvx.SolverStatus.SUBOPTIMAL
        # Infeasible
        assert backend._interpret_status(-200) == nvx.SolverStatus.INFEASIBLE
        # Unbounded
        assert backend._interpret_status(-300) == nvx.SolverStatus.UNBOUNDED
        # Iteration limit
        assert backend._interpret_status(-400) == nvx.SolverStatus.MAX_ITERATIONS
        # Numerical errors
        assert backend._interpret_status(-500) == nvx.SolverStatus.NUMERICAL_ERROR
