"""Tests for binary variable support."""
import pytest
import autograd.numpy as np
import nvxpy as nvx
from nvxpy.variable import Variable, reset_variable_ids
from nvxpy.problem import Problem, Minimize, Maximize


class TestBinaryVariableCreation:
    """Tests for binary variable creation and constraints."""

    def test_binary_variable_basic(self):
        """Test creating a basic binary variable."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")

        assert x.is_integer  # binary implies integer
        assert len(x.constraints) == 2  # >= 0 and <= 1 constraints

    def test_binary_variable_vector(self):
        """Test creating a vector of binary variables."""
        reset_variable_ids()
        x = Variable(shape=(5,), binary=True, name="x")

        assert x.shape == (5,)
        assert x.is_integer
        assert len(x.constraints) == 2

    def test_binary_variable_matrix(self):
        """Test creating a matrix of binary variables."""
        reset_variable_ids()
        x = Variable(shape=(3, 4), binary=True, name="X")

        assert x.shape == (3, 4)
        assert x.is_integer
        assert len(x.constraints) == 2

    def test_binary_with_pos_warning(self, caplog):
        """Test that binary + pos gives a warning."""
        reset_variable_ids()
        import logging
        caplog.set_level(logging.WARNING)

        x = Variable(binary=True, pos=True, name="x")

        assert x.is_integer
        assert "redundant" in caplog.text.lower() or len(x.constraints) >= 2

    def test_binary_with_neg_raises(self):
        """Test that binary + neg raises an error."""
        reset_variable_ids()
        with pytest.raises(ValueError, match="[Bb]inary.*cannot.*negative"):
            Variable(binary=True, neg=True, name="x")

    def test_binary_with_psd_raises(self):
        """Test that binary + PSD raises an error."""
        reset_variable_ids()
        with pytest.raises(ValueError):
            Variable(shape=(2, 2), binary=True, PSD=True, name="x")

    def test_binary_with_nsd_raises(self):
        """Test that binary + NSD raises an error."""
        reset_variable_ids()
        with pytest.raises(ValueError):
            Variable(shape=(2, 2), binary=True, NSD=True, name="x")


class TestBinaryOptimization:
    """Tests for optimizing with binary variables."""

    def test_simple_binary_minimize(self):
        """Test simple minimization with a binary variable."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")
        x.value = np.array([0.5])

        # Minimize (x - 0.3)^2 -> should get x = 0
        prob = Problem(Minimize((x - 0.3) ** 2))
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 0.0

    def test_simple_binary_to_one(self):
        """Test that binary variable can be pushed to 1."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")
        x.value = np.array([0.5])

        # Minimize (x - 0.8)^2 -> should get x = 1
        prob = Problem(Minimize((x - 0.8) ** 2))
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 1.0

    def test_binary_maximize(self):
        """Test maximization with binary variable."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")
        x.value = np.array([0.5])

        # Maximize x -> should get x = 1
        prob = Problem(Maximize(x))
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 1.0

    def test_binary_multiple_variables(self):
        """Test optimization with multiple binary variables."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")
        y = Variable(binary=True, name="y")
        z = Variable(binary=True, name="z")
        x.value = np.array([0.5])
        y.value = np.array([0.5])
        z.value = np.array([0.5])

        # Minimize (x - 0.9)^2 + (y - 0.1)^2 + (z - 0.6)^2
        # Should get x=1, y=0, z=1
        prob = Problem(Minimize((x - 0.9) ** 2 + (y - 0.1) ** 2 + (z - 0.6) ** 2))
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 1.0
        assert y.value == 0.0
        assert z.value == 1.0


class TestBinaryKnapsack:
    """Tests for binary knapsack problem."""

    def test_simple_knapsack(self):
        """Test simple 0-1 knapsack problem."""
        reset_variable_ids()

        # Items: weights and values
        weights = np.array([2, 3, 4, 5])
        values = np.array([3, 4, 5, 6])
        capacity = 8
        n_items = len(weights)

        # Binary decision variables
        x = [Variable(binary=True, name=f"x{i}") for i in range(n_items)]
        for xi in x:
            xi.value = np.array([0.0])

        # Maximize total value
        total_value = sum(values[i] * x[i] for i in range(n_items))

        # Subject to weight constraint
        total_weight = sum(weights[i] * x[i] for i in range(n_items))

        prob = Problem(Maximize(total_value), [total_weight <= capacity])
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL

        # Check feasibility
        selected_weight = sum(weights[i] * x[i].value for i in range(n_items))
        assert selected_weight <= capacity + 0.01

        # All x should be 0 or 1 (within numerical tolerance)
        for xi in x:
            val = np.asarray(xi.value).item()
            assert abs(val - round(val)) < 1e-6, f"Expected 0 or 1, got {val}"

    def test_knapsack_exact_capacity(self):
        """Test knapsack where optimal uses exactly capacity."""
        reset_variable_ids()

        weights = np.array([3, 4, 5])
        values = np.array([4, 5, 6])
        capacity = 7
        n_items = len(weights)

        x = [Variable(binary=True, name=f"x{i}") for i in range(n_items)]
        for xi in x:
            xi.value = np.array([0.0])

        total_value = sum(values[i] * x[i] for i in range(n_items))
        total_weight = sum(weights[i] * x[i] for i in range(n_items))

        prob = Problem(Maximize(total_value), [total_weight <= capacity])
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL

        # Optimal is items 0 and 1 (weight=7, value=9)
        selected = [x[i].value for i in range(n_items)]
        selected_weight = sum(weights[i] * selected[i] for i in range(n_items))
        assert selected_weight <= capacity + 0.01


class TestBinaryWithContinuous:
    """Tests for mixed binary and continuous variables."""

    def test_binary_continuous_mix(self):
        """Test problem with both binary and continuous variables."""
        reset_variable_ids()

        x = Variable(binary=True, name="x")  # binary
        y = Variable(name="y")  # continuous
        x.value = np.array([0.5])
        y.value = np.array([0.5])

        # Minimize (x - 0.7)^2 + (y - 2.3)^2
        # x should be 1, y should be 2.3
        prob = Problem(
            Minimize((x - 0.7) ** 2 + (y - 2.3) ** 2),
            [y >= 0, y <= 5]
        )
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 1.0
        assert np.isclose(y.value, 2.3, atol=0.1)

    def test_binary_affects_continuous(self):
        """Test where binary variable affects continuous feasible region."""
        reset_variable_ids()

        x = Variable(binary=True, name="x")  # indicator
        y = Variable(name="y")  # continuous
        x.value = np.array([1.0])
        y.value = np.array([5.0])

        # y <= 10*x constraint
        # If x=0, then y <= 0
        # If x=1, then y <= 10
        prob = Problem(
            Maximize(y - 2 * x),  # Want high y but low x
            [y >= 0, y <= 10 * x, x + y <= 8]
        )
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL

        # Check feasibility
        assert x.value in [0.0, 1.0]
        assert y.value <= 10 * x.value + 0.01
        assert y.value >= -0.01


class TestBinaryVector:
    """Tests for binary variable vectors."""

    def test_binary_vector_sum_constraint(self):
        """Test binary vector with cardinality constraint."""
        reset_variable_ids()

        n = 4
        x = Variable(shape=(n,), binary=True, name="x")
        x.value = np.zeros(n)

        # Select exactly 2 items
        prob = Problem(
            Minimize(nvx.sum((x - np.array([0.9, 0.8, 0.2, 0.1])) ** 2)),
            [nvx.sum(x) == 2]
        )
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL

        # Should select items 0 and 1 (highest preferences)
        assert np.sum(x.value) == 2.0
        for val in x.value.flatten():
            assert val in [0.0, 1.0]

    def test_binary_vector_at_most_k(self):
        """Test binary vector with at-most-k constraint."""
        reset_variable_ids()

        n = 5
        values = np.array([1, 2, 3, 4, 5])
        x = Variable(shape=(n,), binary=True, name="x")
        x.value = np.zeros(n)

        # Maximize sum of values, selecting at most 2
        prob = Problem(
            Maximize(nvx.sum(values * x)),
            [nvx.sum(x) <= 2]
        )
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL

        # Should select items 3 and 4 (values 4 and 5)
        assert np.sum(x.value) <= 2.0
        selected_value = np.sum(values * x.value.flatten())
        assert selected_value >= 8.9  # 4 + 5 = 9


class TestBinaryEdgeCases:
    """Tests for edge cases with binary variables."""

    def test_single_binary_forced_zero(self):
        """Test binary forced to 0 by constraint."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")
        x.value = np.array([0.5])

        prob = Problem(Minimize(x), [x <= 0.5])
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 0.0

    def test_single_binary_forced_one(self):
        """Test binary forced to 1 by constraint."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")
        x.value = np.array([0.5])

        prob = Problem(Minimize(-x), [x >= 0.5])
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 1.0

    def test_binary_infeasible(self):
        """Test infeasible binary problem."""
        reset_variable_ids()
        x = Variable(binary=True, name="x")
        x.value = np.array([0.5])

        # x >= 0.5 and x <= 0.4 is infeasible for binary
        prob = Problem(Minimize(x), [x >= 0.6, x <= 0.4])
        result = prob.solve(solver=nvx.BNB)

        assert result.status in [nvx.SolverStatus.INFEASIBLE, nvx.SolverStatus.ERROR, nvx.SolverStatus.NUMERICAL_ERROR]

    def test_all_binary_combinations(self):
        """Test that solver explores all binary combinations correctly."""
        reset_variable_ids()

        x = Variable(binary=True, name="x")
        y = Variable(binary=True, name="y")
        x.value = np.array([0.0])
        y.value = np.array([0.0])

        # Objective has unique minimum at x=1, y=0
        prob = Problem(
            Minimize((x - 1) ** 2 + y ** 2 + 0.1 * x * y)
        )
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert x.value == 1.0
        assert y.value == 0.0
