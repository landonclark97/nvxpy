"""
Tests designed to increase coverage of low-coverage modules.

This file consolidates tests for:
- bnb/branching.py (strong, reliability, pseudocost branching)
- bnb/heuristics.py (feasibility pump, round-and-fix)
- bnb/cuts.py (OA cut generation and pruning)
- discrete_set.py (DiscreteRanges, edge cases)
- atoms (axis parameter, edge cases)

Note: Discrete set constraints (x ^ [values]) are now reformulated to
binary indicator variables during Problem construction.
"""

import numpy as np
import pytest

import nvxpy as nvx
from nvxpy.variable import Variable
from nvxpy.problem import Problem, Minimize
from nvxpy.sets.discrete_set import (
    DiscreteSet,
    DiscreteRanges,
    Range,
    _coerce_to_discrete_set,
)
from nvxpy.solvers.bnb.cuts import OACut, generate_oa_cuts, prune_cut_pool
from nvxpy.solvers.bnb.node import PseudocostData
from nvxpy.solvers.bnb.branching import (
    most_fractional_branching,
    pseudocost_branching,
    _fractionality_score,
)


class TestDiscreteRanges:
    """Tests for DiscreteRanges set type."""

    def test_basic_ranges(self):
        """Test basic DiscreteRanges creation."""
        dr = DiscreteRanges([[0, 5], [10, 15]])
        assert len(dr.ranges) == 2
        assert dr.num_branches == 2
        assert 3 in dr
        assert 12 in dr
        assert 7 not in dr

    def test_overlapping_ranges_merge(self):
        """Test that overlapping ranges are merged."""
        dr = DiscreteRanges([[0, 5], [4, 10], [20, 25]])
        assert len(dr.ranges) == 2  # First two should merge
        assert dr.ranges[0].lb == 0
        assert dr.ranges[0].ub == 10

    def test_adjacent_ranges_merge(self):
        """Test that adjacent ranges within tolerance are merged."""
        dr = DiscreteRanges([[0, 5], [5.0001, 10]], tolerance=0.01)
        assert len(dr.ranges) == 1

    def test_nearest(self):
        """Test nearest value computation."""
        dr = DiscreteRanges([[0, 5], [10, 15]])
        assert dr.nearest(3) == 3  # Inside range
        assert dr.nearest(7) == 5  # Closest to first range upper
        assert dr.nearest(8) == 10  # Closest to second range lower
        assert dr.nearest(-5) == 0  # Below all ranges

    def test_ranges_below_above(self):
        """Test ranges_below and ranges_above."""
        dr = DiscreteRanges([[0, 5], [10, 15], [20, 25]])

        below = dr.ranges_below(12)
        assert len(below) == 2  # [0,5] and truncated [10,12-tol]

        above = dr.ranges_above(12)
        assert len(above) == 2  # truncated [12+tol, 15] and [20,25]

    def test_bounds(self):
        """Test overall bounds."""
        dr = DiscreteRanges([[5, 10], [0, 3], [20, 25]])
        lb, ub = dr.bounds()
        assert lb == 0
        assert ub == 25

    def test_invalid_range(self):
        """Test error on invalid range."""
        with pytest.raises(ValueError):
            Range(10, 5)  # lb > ub

    def test_empty_ranges(self):
        """Test error on empty ranges."""
        with pytest.raises(ValueError):
            DiscreteRanges([])

    def test_invalid_range_format(self):
        """Test error on invalid range format."""
        with pytest.raises(ValueError):
            DiscreteRanges([[1, 2, 3]])  # Too many elements


class TestDiscreteSetEdgeCases:
    """Edge cases for DiscreteSet."""

    def test_duplicate_removal(self):
        """Test that duplicates are removed within tolerance."""
        ds = DiscreteSet([1.0, 1.0001, 2.0], tolerance=0.01)
        assert len(ds) == 2

    def test_values_below_above(self):
        """Test values_below and values_above."""
        ds = DiscreteSet([1, 3, 5, 7, 9])
        assert ds.values_below(5) == (1, 3)
        assert ds.values_above(5) == (7, 9)

    def test_iteration(self):
        """Test iteration over set."""
        ds = DiscreteSet([3, 1, 2])
        assert list(ds) == [1, 2, 3]  # Should be sorted

    def test_repr(self):
        """Test string representation."""
        ds = DiscreteSet([1, 2, 3])
        assert "1" in repr(ds)


class TestCoerceToDiscreteSet:
    """Tests for _coerce_to_discrete_set helper."""

    def test_passthrough_discrete_set(self):
        """Test that DiscreteSet passes through."""
        ds = DiscreteSet([1, 2, 3])
        assert _coerce_to_discrete_set(ds) is ds

    def test_passthrough_discrete_ranges(self):
        """Test that DiscreteRanges passes through."""
        dr = DiscreteRanges([[0, 5]])
        assert _coerce_to_discrete_set(dr) is dr

    def test_list_to_discrete_set(self):
        """Test list of scalars to DiscreteSet."""
        result = _coerce_to_discrete_set([1, 2, 3])
        assert isinstance(result, DiscreteSet)

    def test_list_to_discrete_ranges(self):
        """Test list of ranges to DiscreteRanges."""
        result = _coerce_to_discrete_set([[0, 5], [10, 15]])
        assert isinstance(result, DiscreteRanges)

    def test_mixed_error(self):
        """Test error on mixed scalars and ranges."""
        with pytest.raises(ValueError, match="Cannot mix"):
            _coerce_to_discrete_set([1, [0, 5]])

    def test_invalid_type(self):
        """Test error on invalid type."""
        with pytest.raises(TypeError):
            _coerce_to_discrete_set("not a list")


class TestOACuts:
    """Tests for OA cut generation and management."""

    def test_generate_cuts_simple(self):
        """Test OA cut generation from simple constraint."""

        # Create a simple constraint: x >= 1 (as x - 1 >= 0)
        def con_fun(x):
            return np.array([x[0] - 1])

        def con_jac(x):
            return np.array([[1.0]])

        cons = [{"fun": con_fun, "jac": con_jac, "type": "ineq"}]
        x = np.array([2.0])

        cuts = generate_oa_cuts(x, cons)
        assert len(cuts) == 1
        assert cuts[0].coefficients[0] == 1.0

    def test_generate_cuts_equality(self):
        """Test OA cut generation marks equality constraints."""

        def con_fun(x):
            return np.array([x[0] - 1])

        def con_jac(x):
            return np.array([[1.0]])

        cons = [{"fun": con_fun, "jac": con_jac, "type": "eq"}]
        x = np.array([2.0])

        cuts = generate_oa_cuts(x, cons)
        assert len(cuts) == 1
        assert cuts[0].is_equality

    def test_prune_cuts_no_pruning_needed(self):
        """Test prune_cut_pool when no pruning needed."""
        cuts = [OACut(np.array([1.0]), 0.0) for _ in range(5)]
        result = prune_cut_pool(cuts, max_cuts=10, max_age=100)
        assert len(result) == 5

    def test_prune_cuts_remove_old(self):
        """Test that old inactive cuts are removed."""
        cuts = [
            OACut(np.array([1.0]), 0.0, age=200, times_active=0),
            OACut(np.array([2.0]), 0.0, age=10, times_active=0),
        ]
        result = prune_cut_pool(cuts, max_cuts=10, max_age=100)
        assert len(result) == 1
        assert result[0].coefficients[0] == 2.0

    def test_prune_cuts_keep_active(self):
        """Test that active cuts are kept even if old."""
        cuts = [
            OACut(np.array([1.0]), 0.0, age=200, times_active=5),
            OACut(np.array([2.0]), 0.0, age=200, times_active=0),
        ]
        result = prune_cut_pool(cuts, max_cuts=10, max_age=100)
        assert len(result) == 1
        assert result[0].coefficients[0] == 1.0

    def test_prune_cuts_limit(self):
        """Test that prune respects max_cuts limit."""
        cuts = [OACut(np.array([float(i)]), 0.0, age=i) for i in range(20)]
        result = prune_cut_pool(cuts, max_cuts=5, max_age=100)
        assert len(result) == 5


class TestBranchingHelpers:
    """Tests for branching helper functions."""

    def test_fractionality_score_standard(self):
        """Test fractionality score for standard integers."""
        # Most fractional at 0.5
        score_half = _fractionality_score(2.5)
        score_near = _fractionality_score(2.1)
        assert score_half < score_near  # Lower score = more fractional

    def test_most_fractional_branching(self):
        """Test most fractional variable selection."""
        violations = [(0, 2.5), (1, 2.1), (2, 2.9)]  # 0 is most fractional
        idx, val = most_fractional_branching(violations)
        assert idx == 0
        assert val == 2.5

    def test_pseudocost_branching(self):
        """Test pseudocost branching."""
        violations = [(0, 2.5), (1, 2.5)]
        pseudocosts = {
            0: PseudocostData(down_cost=1.0, up_cost=1.0, down_count=5, up_count=5),
            1: PseudocostData(down_cost=10.0, up_cost=10.0, down_count=5, up_count=5),
        }
        idx, val = pseudocost_branching(violations, pseudocosts)
        assert idx == 1  # Higher pseudocost score


class TestBnBWithDiscreteRanges:
    """Tests for B&B with DiscreteRanges constraints."""

    def test_bnb_discrete_ranges_simple(self):
        """Test B&B with simple discrete ranges."""
        nvx.reset_variable_ids()
        x = Variable(name="x")
        x.value = np.array([0.0])

        # x must be in [0, 2] or [5, 7]
        # Minimize (x - 6)^2 should give x in [5, 7]
        prob = Problem(Minimize((x - 6) ** 2), [x ^ [[0, 2], [5, 7]]])
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL
        val = np.asarray(x.value).item()
        assert 5 <= val <= 7

    def test_bnb_discrete_ranges_multiple(self):
        """Test B&B with multiple range constraints."""
        nvx.reset_variable_ids()
        x = Variable(name="x")
        y = Variable(name="y")
        x.value = np.array([0.0])
        y.value = np.array([0.0])

        prob = Problem(
            Minimize((x - 3) ** 2 + (y - 8) ** 2),
            [
                x ^ [[0, 2], [4, 6]],
                y ^ [[0, 5], [7, 10]],
            ],
        )
        result = prob.solve(solver=nvx.BNB)

        assert result.status == nvx.SolverStatus.OPTIMAL


class TestBnBFeasibilityPump:
    """Tests for B&B with feasibility pump heuristic."""

    def test_bnb_feasibility_pump(self):
        """Test B&B with feasibility pump enabled."""
        nvx.reset_variable_ids()
        x = Variable(integer=True, name="x")
        y = Variable(integer=True, name="y")
        x.value = np.array([0.0])
        y.value = np.array([0.0])

        prob = Problem(
            Minimize((x - 3) ** 2 + (y - 4) ** 2), [x >= 0, y >= 0, x <= 10, y <= 10]
        )
        result = prob.solve(
            solver=nvx.BNB,
            solver_options={
                "fp_max_iterations": 5,
                "use_heuristics": True,
            },
        )

        assert result.status == nvx.SolverStatus.OPTIMAL


class TestAtomsAxisParameter:
    """Tests for atoms with axis parameter (coverage for shape logic)."""

    def test_sum_with_axis(self):
        """Test sum with axis parameter."""
        x = Variable((3, 4), name="x")
        s = nvx.sum(x, axis=0)
        assert s.shape == (4,)

        s = nvx.sum(x, axis=1)
        assert s.shape == (3,)

        s = nvx.sum(x, axis=-1)  # Negative axis
        assert s.shape == (3,)

    def test_amax_with_axis(self):
        """Test amax with axis parameter."""
        x = Variable((3, 4), name="x")
        m = nvx.amax(x, axis=0)
        assert m.shape == (4,)

        m = nvx.amax(x, axis=1)
        assert m.shape == (3,)

    def test_amin_with_axis(self):
        """Test amin with axis parameter."""
        x = Variable((3, 4), name="x")
        m = nvx.amin(x, axis=0)
        assert m.shape == (4,)

        m = nvx.amin(x, axis=1)
        assert m.shape == (3,)

    def test_axis_out_of_bounds(self):
        """Test that out of bounds axis raises error."""
        x = Variable((3,), name="x")
        with pytest.raises(ValueError, match="out of bounds"):
            _ = nvx.sum(x, axis=5).shape

        with pytest.raises(ValueError, match="out of bounds"):
            _ = nvx.amax(x, axis=2).shape

        with pytest.raises(ValueError, match="out of bounds"):
            _ = nvx.amin(x, axis=-3).shape


class TestAtomsCurvature:
    """Additional curvature tests for atoms."""

    def test_amax_concave_unknown(self):
        """Test amax of concave expression is unknown."""
        x = Variable((3,), name="x")
        # sqrt is concave, so amax(sqrt(x)) should be unknown
        expr = nvx.amax(nvx.sqrt(x))
        assert expr.curvature == nvx.Curvature.UNKNOWN

    def test_amin_convex_unknown(self):
        """Test amin of convex expression is unknown."""
        x = Variable((3,), name="x")
        # exp is convex, so amin(exp(x)) should be unknown
        expr = nvx.amin(nvx.exp(x))
        assert expr.curvature == nvx.Curvature.UNKNOWN


class TestBnBStrongBranchingDiscrete:
    """Tests for strong branching with discrete variables."""

    def test_strong_branching_discrete_values(self):
        """Test strong branching with discrete set."""
        nvx.reset_variable_ids()
        x = Variable(integer=True, name="x")
        y = Variable(integer=True, name="y")
        x.value = np.array([0.0])
        y.value = np.array([0.0])

        prob = Problem(
            Minimize((x - 2.7) ** 2 + (y - 3.3) ** 2),
            [x ^ [1, 2, 3, 4], y ^ [1, 2, 3, 4, 5]],
        )
        result = prob.solve(
            solver=nvx.BNB, solver_options={"branching": "strong", "strong_limit": 5}
        )

        assert result.status == nvx.SolverStatus.OPTIMAL


class TestBnBReliabilityBranchingDiscrete:
    """Tests for reliability branching with discrete variables."""

    def test_reliability_with_discrete(self):
        """Test reliability branching transitions to pseudocost."""
        nvx.reset_variable_ids()
        x = Variable(integer=True, name="x")
        y = Variable(integer=True, name="y")
        z = Variable(integer=True, name="z")
        x.value = np.array([0.0])
        y.value = np.array([0.0])
        z.value = np.array([0.0])

        prob = Problem(
            Minimize((x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2),
            [
                x >= 0,
                y >= 0,
                z >= 0,
                x <= 5,
                y <= 5,
                z <= 5,
            ],
        )
        result = prob.solve(
            solver=nvx.BNB,
            solver_options={
                "branching": "reliability",
                "reliability_limit": 1,
                "strong_limit": 2,
            },
        )

        assert result.status == nvx.SolverStatus.OPTIMAL


class TestRangeContainment:
    """Tests for Range containment checks."""

    def test_range_contains(self):
        """Test Range __contains__ method."""
        r = Range(0, 10)
        assert 5 in r
        assert 0 in r
        assert 10 in r
        assert -1 not in r
        assert 11 not in r

    def test_range_repr(self):
        """Test Range string representation."""
        r = Range(0, 10)
        assert "[0, 10]" in repr(r)


class TestFunctionDecorator:
    """Tests for the @nvx.function decorator."""

    def test_decorator_without_args(self):
        """Test @nvx.function without parentheses."""

        @nvx.function
        def my_func(x):
            return x[0] ** 2 + x[1] ** 2

        assert isinstance(my_func, nvx.Function)

        x = nvx.Variable((2,))
        expr = my_func(x)
        assert expr.op == "func"

    def test_decorator_with_args(self):
        """Test @nvx.function with keyword arguments."""

        @nvx.function(jac="autograd", shape=(1,))
        def my_func(x):
            return x[0] ** 2 + x[1] ** 2

        assert isinstance(my_func, nvx.Function)
        assert my_func._shape == (1,)

    def test_decorator_in_problem(self):
        """Test decorated function in an optimization problem."""

        @nvx.function
        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        x = nvx.Variable((2,), name="x")
        x.value = [0.0, 0.0]

        prob = nvx.Problem(nvx.Minimize(rosenbrock(x)))
        result = prob.solve()

        assert result.status == nvx.SolverStatus.OPTIMAL
        assert abs(x.value[0] - 1.0) < 0.1
        assert abs(x.value[1] - 1.0) < 0.1
