"""Tests for the Branch-and-Bound MINLP solver."""
import autograd.numpy as np
import nvxpy as nvx
from nvxpy.variable import Variable
from nvxpy.problem import Problem, Minimize


def test_bnb_simple_integer():
    """Test B&B with simple integer variable."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    # Minimize (x - 2.7)^2, should get x = 3
    prob = Problem(Minimize((x - 2.7) ** 2), [x >= 0, x <= 10])
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 3.0, atol=0.1)


def test_bnb_multiple_integers():
    """Test B&B with multiple integer variables."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    # Minimize (x - 1.5)^2 + (y - 2.5)^2
    prob = Problem(
        Minimize((x - 1.5) ** 2 + (y - 2.5) ** 2),
        [x >= 0, y >= 0, x <= 5, y <= 5]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    # x should be 1 or 2, y should be 2 or 3
    assert x.value in [1.0, 2.0]
    assert y.value in [2.0, 3.0]


def test_bnb_depth_first():
    """Test B&B with depth-first node selection."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 5) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"node_selection": "depth_first"}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 5.0, atol=0.1)


def test_bnb_hybrid():
    """Test B&B with hybrid node selection."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 5) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"node_selection": "hybrid"}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 5.0, atol=0.1)


def test_bnb_pseudocost_branching():
    """Test B&B with pseudocost branching."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2),
        [x >= 0, y >= 0, x <= 10, y <= 10]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"branching": "pseudocost"}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_strong_branching():
    """Test B&B with strong branching."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"branching": "strong", "strong_limit": 2}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 3.0, atol=0.1)


def test_bnb_reliability_branching():
    """Test B&B with reliability branching."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"branching": "reliability", "reliability_limit": 2}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_with_heuristics():
    """Test B&B with heuristics enabled."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 5) ** 2 + (y - 5) ** 2),
        [x >= 0, y >= 0, x + y <= 15]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"use_heuristics": True}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_with_oa_cuts():
    """Test B&B with outer approximation cuts."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"use_oa_cuts": True}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_verbose():
    """Test B&B with verbose output."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 5])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"verbose": True}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_max_nodes():
    """Test B&B with max_nodes limit."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 50) ** 2 + (y - 50) ** 2),
        [x >= 0, y >= 0, x <= 100, y <= 100]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"max_nodes": 5}
    )

    # May not be optimal due to node limit
    assert result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]


def test_bnb_gap_tolerance():
    """Test B&B with custom gap tolerance."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"rel_gap": 0.1, "abs_gap": 1.0}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_with_continuous_vars():
    """Test B&B with mixed integer and continuous variables."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(name="y")  # continuous
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    # Minimize (x - 2.5)^2 + (y - 1.7)^2
    prob = Problem(
        Minimize((x - 2.5) ** 2 + (y - 1.7) ** 2),
        [x >= 0, y >= 0, x <= 5, y <= 5]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    # x should be 2 or 3, y should be close to 1.7
    assert x.value in [2.0, 3.0]
    assert np.isclose(y.value, 1.7, atol=0.1)


def test_bnb_with_constraints():
    """Test B&B with additional constraints."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    # Minimize x + y subject to x + y >= 5
    prob = Problem(
        Minimize(x + y),
        [x >= 0, y >= 0, x + y >= 5]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value + y.value >= 5 - 0.01


def test_bnb_compile():
    """Test B&B with expression compilation."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(solver=nvx.BNB, compile=True)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 3.0, atol=0.1)


def test_bnb_discrete_binary():
    """Test B&B with binary-like discrete constraint."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    # x in {0, 1} - binary
    prob = Problem(
        Minimize((x - 0.7) ** 2),
        [x ^ [0, 1]]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value == 1.0


def test_bnb_nonlinear_constraint():
    """Test B&B with nonlinear constraints."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(name="y")
    x.value = np.array([1.0])
    y.value = np.array([1.0])

    # Minimize x + y subject to x^2 + y^2 <= 10
    prob = Problem(
        Minimize(x + y),
        [x >= 0, y >= 0, x ** 2 + y ** 2 <= 10]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value ** 2 + y.value ** 2 <= 10 + 0.1


def test_bnb_time_limit():
    """Test B&B with time limit."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 50) ** 2 + (y - 50) ** 2),
        [x >= 0, y >= 0, x <= 100, y <= 100]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"max_time": 0.5}  # 0.5 second limit
    )

    # Should complete (may or may not be optimal)
    assert result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]


def test_bnb_int_tolerance():
    """Test B&B with custom integer tolerance."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"int_tol": 0.01}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_nlp_method():
    """Test B&B with different NLP solver method."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    # Use COBYLA which supports constraints (derivative-free)
    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"nlp_method": "COBYLA"}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_result_stats():
    """Test that B&B returns useful stats."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(Minimize((x - 3) ** 2), [x >= 0, x <= 10])
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert result.stats is not None
    assert "B&B" in result.stats.solver_name


def test_bnb_verbose_with_discrete():
    """Test B&B verbose output with discrete constraints."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 5) ** 2),
        [x ^ [1, 3, 5, 7, 9]]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"verbose": True}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value == 5.0


def test_bnb_verbose_with_time_limit():
    """Test B&B verbose output when time limit is hit."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 50) ** 2 + (y - 50) ** 2),
        [x >= 0, y >= 0, x <= 100, y <= 100]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"verbose": True, "max_time": 0.01}
    )

    # Just verify it doesn't crash
    assert result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]


def test_bnb_verbose_with_node_limit():
    """Test B&B verbose output when node limit is hit."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 50) ** 2 + (y - 50) ** 2),
        [x >= 0, y >= 0, x <= 100, y <= 100]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"verbose": True, "max_nodes": 3}
    )

    assert result.status in [nvx.SolverStatus.OPTIMAL, nvx.SolverStatus.SUBOPTIMAL]


def test_bnb_depth_first_with_more_nodes():
    """Test B&B depth-first with enough nodes to exercise loop."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2),
        [x >= 0, y >= 0, x <= 10, y <= 10]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"node_selection": "depth_first"}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_hybrid_with_more_nodes():
    """Test B&B hybrid selection with many nodes to hit best-first branch."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    z = Variable(integer=True, name="z")
    x.value = np.array([0.0])
    y.value = np.array([0.0])
    z.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2 + (z - 2) ** 2),
        [x >= 0, y >= 0, z >= 0, x <= 10, y <= 10, z <= 10]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"node_selection": "hybrid"}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_pseudocost_with_discrete():
    """Test B&B pseudocost branching with discrete variables."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2),
        [x ^ [1, 2, 3, 4, 5], y ^ [1, 2, 3, 4, 5, 6, 7]]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"branching": "pseudocost"}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value == 3.0
    assert y.value == 4.0


def test_bnb_strong_with_discrete():
    """Test B&B strong branching with discrete variables."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 3) ** 2),
        [x ^ [1, 2, 3, 4, 5]]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"branching": "strong", "strong_limit": 3}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value == 3.0


def test_bnb_reliability_with_multiple_vars():
    """Test B&B reliability branching with multiple variables."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2),
        [x >= 0, y >= 0, x <= 10, y <= 10]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"branching": "reliability", "reliability_limit": 1}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_heuristics_with_many_nodes():
    """Test B&B heuristics with enough nodes to trigger mid-solve heuristics."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    z = Variable(integer=True, name="z")
    x.value = np.array([5.0])
    y.value = np.array([5.0])
    z.value = np.array([5.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2 + (z - 2) ** 2),
        [x >= 0, y >= 0, z >= 0, x <= 10, y <= 10, z <= 10, x + y + z <= 15]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"use_heuristics": True, "verbose": True}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_oa_cuts_with_constraints():
    """Test B&B with OA cuts and multiple constraints."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(name="y")
    x.value = np.array([1.0])
    y.value = np.array([1.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 2) ** 2),
        [x >= 0, y >= 0, x + y <= 6, x <= 5]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"use_oa_cuts": True}
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_many_discrete_values():
    """Test B&B with many discrete values to exercise fractionality scoring."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([0.0])

    # Many discrete values
    prob = Problem(
        Minimize((x - 7.3) ** 2),
        [x ^ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert x.value == 7.0


def test_bnb_gap_early_termination():
    """Test B&B with gap tolerance for early termination."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(integer=True, name="y")
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 5) ** 2 + (y - 5) ** 2),
        [x >= 0, y >= 0, x <= 10, y <= 10]
    )
    result = prob.solve(
        solver=nvx.BNB,
        solver_options={"rel_gap": 0.5, "verbose": True}  # Large gap for early stop
    )

    assert result.status == nvx.SolverStatus.OPTIMAL


def test_bnb_equality_constraint():
    """Test B&B with equality constraint."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    y = Variable(name="y")
    x.value = np.array([2.0])
    y.value = np.array([3.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2),
        [x + y == 7, x >= 0, y >= 0]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value + y.value, 7.0, atol=0.1)


def test_bnb_infeasible():
    """Test B&B with infeasible problem."""
    nvx.reset_variable_ids()
    x = Variable(integer=True, name="x")
    x.value = np.array([5.0])

    # Infeasible: x >= 10 and x <= 5
    prob = Problem(
        Minimize(x ** 2),
        [x >= 10, x <= 5]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.INFEASIBLE


def test_bnb_no_integer_vars():
    """Test B&B falls back to NLP when no integer variables."""
    nvx.reset_variable_ids()
    x = Variable(name="x")  # continuous
    y = Variable(name="y")  # continuous
    x.value = np.array([0.0])
    y.value = np.array([0.0])

    prob = Problem(
        Minimize((x - 3) ** 2 + (y - 4) ** 2),
        [x >= 0, y >= 0]
    )
    result = prob.solve(solver=nvx.BNB)

    assert result.status == nvx.SolverStatus.OPTIMAL
    assert np.isclose(x.value, 3.0, atol=0.1)
    assert np.isclose(y.value, 4.0, atol=0.1)
