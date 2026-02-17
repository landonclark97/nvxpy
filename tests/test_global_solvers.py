"""Tests for global scipy optimizer backends."""

import numpy as np
import pytest

import nvxpy as nvx
from nvxpy.solvers import SolverStatus


class TestDifferentialEvolution:
    """Tests for differential_evolution solver."""

    def test_unconstrained_quadratic(self):
        """Simple quadratic finds global minimum."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum(x**2)

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5])
        result = prob.solve(solver=nvx.DIFF_EVOLUTION, solver_options={"seed": 0})

        assert result.status == SolverStatus.OPTIMAL
        assert np.allclose(x.value, [0, 0], atol=1e-4)

    def test_with_nonlinear_constraint(self):
        """Quadratic with nonlinear constraint."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum((x - 1) ** 2)

        # Constrain to unit circle
        constraints = [
            x >= -2,
            x <= 2,
            nvx.sum(x**2) <= 1,
        ]

        prob = nvx.Problem(nvx.Minimize(objective), constraints)
        result = prob.solve(solver=nvx.DIFF_EVOLUTION, solver_options={"seed": 0})

        assert result.status == SolverStatus.OPTIMAL
        # Solution should be on the circle, closest to (1, 1)
        assert np.linalg.norm(x.value) <= 1.0 + 1e-4

    def test_rastrigin(self):
        """Rastrigin function - a classic global optimization test."""
        x = nvx.Variable((2,), name="x")
        objective = 20 + nvx.sum(x**2 - 10 * nvx.cos(2 * np.pi * x))

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5.12, x <= 5.12])
        result = prob.solve(
            solver=nvx.DIFF_EVOLUTION,
            solver_options={"seed": 0, "maxiter": 200},
        )

        assert result.status == SolverStatus.OPTIMAL
        assert np.allclose(x.value, [0, 0], atol=1e-3)
        assert objective.value < 0.01


class TestDualAnnealing:
    """Tests for dual_annealing solver."""

    def test_unconstrained_quadratic(self):
        """Simple quadratic finds global minimum."""
        x = nvx.Variable(name="x")
        objective = (x - 3) ** 2

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -10, x <= 10])
        result = prob.solve(solver=nvx.DUAL_ANNEALING, solver_options={"seed": 0})

        assert result.status == SolverStatus.OPTIMAL
        assert np.isclose(np.asarray(x.value).item(), 3.0, atol=1e-4)

    def test_multimodal(self):
        """Multi-well function escapes local minima."""
        x = nvx.Variable(name="x")
        # Function with wells at x=-2 and x=2, minimum near x=2 is global
        objective = (x - 2) ** 2 * (x + 2) ** 2 + 0.5 * x

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5])
        result = prob.solve(solver=nvx.DUAL_ANNEALING, solver_options={"seed": 42})

        assert result.status == SolverStatus.OPTIMAL
        # Should find one of the minima and complete
        val = np.asarray(x.value).item()
        assert abs(val - 2) < 0.5 or abs(val + 2) < 0.5

    def test_requires_finite_bounds(self):
        """dual_annealing requires finite bounds."""
        x = nvx.Variable(name="x")
        objective = x**2

        prob = nvx.Problem(nvx.Minimize(objective))
        with pytest.raises(ValueError, match="finite bounds"):
            prob.solve(solver=nvx.DUAL_ANNEALING)


class TestSHGO:
    """Tests for shgo (simplicial homology global optimization) solver."""

    def test_unconstrained_quadratic(self):
        """Simple quadratic finds global minimum."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum((x - 1) ** 2)

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5])
        result = prob.solve(solver=nvx.SHGO)

        assert result.status == SolverStatus.OPTIMAL
        assert np.allclose(x.value, [1, 1], atol=1e-4)

    def test_with_constraint(self):
        """SHGO with nonlinear constraint."""
        x = nvx.Variable((2,), name="x")
        # Rosenbrock
        objective = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        constraints = [
            x >= -2,
            x <= 2,
            nvx.sum(x**2) <= 3,
        ]

        prob = nvx.Problem(nvx.Minimize(objective), constraints)
        result = prob.solve(solver=nvx.SHGO, solver_options={"n": 100, "iters": 3})

        assert result.status == SolverStatus.OPTIMAL
        assert np.allclose(x.value, [1, 1], atol=0.1)


class TestBasinhopping:
    """Tests for basinhopping solver."""

    def test_no_bounds_required(self):
        """basinhopping works without explicit bounds."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum((x - 2) ** 2)

        prob = nvx.Problem(nvx.Minimize(objective))
        result = prob.solve(solver=nvx.BASINHOPPING, solver_options={"niter": 10})

        assert result.status == SolverStatus.OPTIMAL
        assert np.allclose(x.value, [2, 2], atol=1e-4)

    def test_with_bounds(self):
        """basinhopping uses L-BFGS-B when bounds are present."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum((x - 2) ** 2)

        prob = nvx.Problem(nvx.Minimize(objective), [x >= 0, x <= 5])
        result = prob.solve(solver=nvx.BASINHOPPING, solver_options={"niter": 10})

        assert result.status == SolverStatus.OPTIMAL
        assert np.allclose(x.value, [2, 2], atol=1e-4)

    def test_does_not_support_constraints(self):
        """basinhopping cannot handle nonlinear constraints."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum(x**2)

        # Nonlinear constraint (not simple bounds)
        constraints = [x >= -5, x <= 5, nvx.sum(x**2) <= 1]

        prob = nvx.Problem(nvx.Minimize(objective), constraints)
        with pytest.raises(ValueError, match="does not support constraints"):
            prob.solve(solver=nvx.BASINHOPPING)

    def test_himmelblau(self):
        """Himmelblau's function has 4 global minima."""
        x = nvx.Variable((2,), name="x")
        objective = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5])
        result = prob.solve(
            solver=nvx.BASINHOPPING, solver_options={"niter": 20, "seed": 0}
        )

        assert result.status == SolverStatus.OPTIMAL
        assert np.isclose(objective.value, 0.0, atol=1e-4)


class TestGlobalSolverEdgeCases:
    """Edge cases and error handling for global solvers."""

    def test_integer_vars_not_supported(self):
        """Global solvers don't support integer variables."""
        x = nvx.Variable(name="x", integer=True)
        prob = nvx.Problem(nvx.Minimize(x**2), [x >= -5, x <= 5])

        with pytest.raises(ValueError, match="[Ii]nteger"):
            prob.solve(solver=nvx.DIFF_EVOLUTION)

    def test_constraint_aware_methods(self):
        """Only diff_evolution and shgo support nonlinear constraints."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum(x**2)
        constraints = [x >= -5, x <= 5, nvx.sum(x**2) <= 1]  # nonlinear

        prob = nvx.Problem(nvx.Minimize(objective), constraints)

        # These should work
        result = prob.solve(solver=nvx.DIFF_EVOLUTION)
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.SUBOPTIMAL)

        result = prob.solve(solver=nvx.SHGO)
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.SUBOPTIMAL)

    def test_simple_bounds_only(self):
        """Solvers that don't support constraints work with simple bounds."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum((x - 1) ** 2)

        # Only simple bounds, no nonlinear constraints
        constraints = [x >= -5, x <= 5]

        prob = nvx.Problem(nvx.Minimize(objective), constraints)

        # dual_annealing should work with simple bounds
        result = prob.solve(solver=nvx.DUAL_ANNEALING, solver_options={"seed": 0})
        assert result.status == SolverStatus.OPTIMAL

        # basinhopping should work too
        result = prob.solve(solver=nvx.BASINHOPPING, solver_options={"niter": 10})
        assert result.status == SolverStatus.OPTIMAL

    def test_solver_options_passed(self):
        """Verify solver options are passed through."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum(x**2)

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5])

        # Test maxiter option
        result = prob.solve(
            solver=nvx.DIFF_EVOLUTION,
            solver_options={"maxiter": 5, "seed": 0},
        )
        # Should complete (may not be optimal with only 5 iterations)
        assert result.status in (SolverStatus.OPTIMAL, SolverStatus.MAX_ITERATIONS)

    def test_compile_flag(self):
        """Test that compile=True works with global solvers."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum(x**2 - 2 * x + 1)

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5], compile=True)
        result = prob.solve(solver=nvx.DIFF_EVOLUTION, solver_options={"seed": 0})

        assert result.status == SolverStatus.OPTIMAL
        assert np.allclose(x.value, [1, 1], atol=1e-4)


class TestGlobalSolverStats:
    """Test that solver statistics are properly returned."""

    def test_stats_populated(self):
        """Solver stats should be populated."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum(x**2)

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5])
        result = prob.solve(solver=nvx.DIFF_EVOLUTION, solver_options={"seed": 0})

        assert result.stats is not None
        assert result.stats.solver_name == "differential_evolution"
        assert result.stats.solve_time is not None
        assert result.stats.solve_time > 0

    def test_raw_result_structure(self):
        """Raw result should contain primary and projection."""
        x = nvx.Variable((2,), name="x")
        objective = nvx.sum(x**2)

        prob = nvx.Problem(nvx.Minimize(objective), [x >= -5, x <= 5])
        result = prob.solve(solver=nvx.DIFF_EVOLUTION, solver_options={"seed": 0})

        assert result.raw_result is not None
        assert "primary" in result.raw_result
        assert "projection" in result.raw_result
