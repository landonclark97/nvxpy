"""
Utility Functions for Branch-and-Bound

This module contains utility functions shared across the B&B implementation,
including bound extraction, constraint handling, and node management.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import autograd.numpy as np
from autograd import jacobian

from ...constants import Curvature
from ..base import ProblemData
from .node import BBNode, DiscreteVarInfo

logger = logging.getLogger(__name__)


def extract_simple_bounds(problem_data: ProblemData) -> Dict[int, Tuple[float, float]]:
    """
    Extract simple variable bounds from constraints.

    Looks for constraints of the form:
    - var >= constant  (lower bound)
    - var <= constant  (upper bound)
    - constant <= var  (lower bound)
    - constant >= var  (upper bound)

    Returns a dict mapping flat index -> (lb, ub).
    """
    from ...variable import Variable

    var_bounds: Dict[int, Tuple[float, float]] = {}
    n_vars = len(problem_data.x0)

    # Initialize with no bounds
    for i in range(n_vars):
        var_bounds[i] = (float("-inf"), float("inf"))

    for constraint in problem_data.constraints:
        if constraint.op not in (">=", "<=", "=="):
            continue

        # Check if this is a simple bound: var op constant or constant op var
        left = constraint.left
        right = constraint.right

        var = None
        const = None
        is_var_on_left = False

        if isinstance(left, Variable) and isinstance(right, (int, float)):
            var = left
            const = float(right)
            is_var_on_left = True
        elif isinstance(right, Variable) and isinstance(left, (int, float)):
            var = right
            const = float(left)
            is_var_on_left = False
        else:
            continue

        if var.name not in problem_data.var_slices:
            continue

        start, end = problem_data.var_slices[var.name]

        # Determine bound type based on operator and which side var is on
        for idx in range(start, end):
            current_lb, current_ub = var_bounds[idx]

            if constraint.op == ">=":
                if is_var_on_left:
                    # var >= const -> lower bound
                    current_lb = max(current_lb, const)
                else:
                    # const >= var -> upper bound
                    current_ub = min(current_ub, const)
            elif constraint.op == "<=":
                if is_var_on_left:
                    # var <= const -> upper bound
                    current_ub = min(current_ub, const)
                else:
                    # const <= var -> lower bound
                    current_lb = max(current_lb, const)
            elif constraint.op == "==":
                # var == const -> fixed
                current_lb = max(current_lb, const)
                current_ub = min(current_ub, const)

            var_bounds[idx] = (current_lb, current_ub)

    # Remove entries with no actual bounds (both inf)
    return {
        idx: bounds for idx, bounds in var_bounds.items()
        if bounds[0] > float("-inf") or bounds[1] < float("inf")
    }


def propagate_discrete_bounds(
    discrete_vars: Dict[int, DiscreteVarInfo],
    simple_bounds: Dict[int, Tuple[float, float]] | None = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Initialize tight bounds from discrete constraints and simple variable bounds.

    For each discrete variable, sets bounds to [min_allowed, max_allowed].
    Also incorporates simple bounds extracted from constraints.
    """
    var_bounds: Dict[int, Tuple[float, float]] = {}

    # First, add simple bounds
    if simple_bounds:
        var_bounds.update(simple_bounds)

    # Then, tighten with discrete bounds (these are typically tighter)
    for idx, dvar in discrete_vars.items():
        # Collect all lower/upper bounds from discrete values and ranges
        lb_candidates = list(dvar.allowed_values)
        ub_candidates = list(dvar.allowed_values)

        # Add range bounds
        for r_lb, r_ub in dvar.allowed_ranges:
            lb_candidates.append(r_lb)
            ub_candidates.append(r_ub)

        if lb_candidates:  # Has at least one value or range
            lb = min(lb_candidates)
            ub = max(ub_candidates)

            # Intersect with existing bounds
            if idx in var_bounds:
                existing_lb, existing_ub = var_bounds[idx]
                lb = max(lb, existing_lb)
                ub = min(ub, existing_ub)

            var_bounds[idx] = (lb, ub)

    return var_bounds


def get_integer_indices(problem_data: ProblemData) -> List[int]:
    """Get flat indices for all integer variable elements."""
    indices = []
    for var_name in problem_data.integer_vars:
        start, end = problem_data.var_slices[var_name]
        indices.extend(range(start, end))
    return indices


def get_warm_start(node: BBNode, problem_data: ProblemData) -> np.ndarray:
    """Get warm start, projected to node bounds."""
    if node.parent_solution is not None:
        x0 = node.parent_solution.copy()
    else:
        x0 = problem_data.x0.copy()

    # Project to node bounds
    for idx, (lb, ub) in node.var_bounds.items():
        x0[idx] = np.clip(x0[idx], lb, ub)

    return x0


def get_scipy_bounds(
    node: BBNode,
    n_vars: int,
) -> List[Tuple[float | None, float | None]]:
    """Get bounds for scipy.optimize.minimize."""
    bounds: List[Tuple[float | None, float | None]] = [(None, None)] * n_vars

    # Apply node-specific bounds for integer variables
    for idx, (lb, ub) in node.var_bounds.items():
        bounds[idx] = (lb, ub)

    return bounds


def get_integer_violations(
    x: np.ndarray,
    int_indices: List[int],
    tol: float,
    discrete_vars: Dict[int, DiscreteVarInfo] | None = None,
    node_bounds: Dict[int, Tuple[float, float]] | None = None,
) -> List[Tuple[int, float]]:
    """Get list of (index, value) for variables violating integrality/discreteness."""
    violations = []

    # Check integer variables
    for idx in int_indices:
        val = x[idx]

        # Check if this index has a discrete constraint
        if discrete_vars and idx in discrete_vars:
            dvar = discrete_vars[idx]
            # Pure discrete values - use standard check
            if not dvar.is_feasible(val):
                violations.append((idx, val))
        else:
            # Standard integer check
            frac = abs(val - round(val))
            if frac > tol:
                violations.append((idx, val))

    # Check pure range variables (DiscreteRanges) - these are NOT in int_indices
    # but still need branching until committed to a single range
    if discrete_vars:
        for idx, dvar in discrete_vars.items():
            if idx in int_indices:
                continue  # Already checked above

            val = x[idx]

            # Get current bounds for this variable
            if node_bounds and idx in node_bounds:
                current_lb, current_ub = node_bounds[idx]
            else:
                current_lb, current_ub = -1e8, 1e8

            # Count how many ranges are still reachable within current bounds
            reachable_ranges = 0
            for r_lb, r_ub in dvar.allowed_ranges:
                # Range overlaps with current bounds?
                if r_lb <= current_ub + tol and r_ub >= current_lb - tol:
                    reachable_ranges += 1

            # If multiple ranges are reachable, we need to branch
            if reachable_ranges > 1:
                violations.append((idx, val))
            # If only one range but value not in it, still a violation
            elif not dvar.is_feasible(val):
                violations.append((idx, val))

    return violations


def extract_discrete_constraints(
    problem_data: ProblemData,
) -> Tuple[Dict[int, DiscreteVarInfo], List]:
    """
    Extract discrete (x ^ [values]) and range (x ^ [[lb, ub], ...]) constraints from problem data.

    Handles both DiscreteSet (discrete values only) and DiscreteRanges
    (continuous ranges for disjunctive constraints).

    Returns:
        Tuple of:
        - Dict mapping flat index -> DiscreteVarInfo
        - List of remaining constraints (non-discrete)
    """
    from ...sets.discrete_set import DiscreteSet, DiscreteRanges
    from ...variable import Variable

    discrete_vars: Dict[int, DiscreteVarInfo] = {}
    remaining_constraints = []

    for constraint in problem_data.constraints:
        if constraint.op == "in" and isinstance(constraint.right, (DiscreteSet, DiscreteRanges)):
            var = constraint.left
            the_set = constraint.right

            # Get the variable name
            if isinstance(var, Variable):
                var_name = var.name
                if var_name in problem_data.var_slices:
                    start, end = problem_data.var_slices[var_name]

                    # Extract values and ranges depending on set type
                    if isinstance(the_set, DiscreteRanges):
                        allowed_values = ()
                        allowed_ranges = tuple((r.lb, r.ub) for r in the_set.ranges)
                    else:
                        # DiscreteSet - only discrete values
                        allowed_values = the_set.values
                        allowed_ranges = ()

                    # For each element of the variable, add discrete info
                    for idx in range(start, end):
                        discrete_vars[idx] = DiscreteVarInfo(
                            var_name=var_name,
                            flat_index=idx,
                            allowed_values=allowed_values,
                            allowed_ranges=allowed_ranges,
                            tolerance=the_set.tolerance,
                        )
        else:
            remaining_constraints.append(constraint)

    return discrete_vars, remaining_constraints


def round_to_discrete(
    x: np.ndarray,
    int_indices: List[int],
    discrete_vars: Dict[int, DiscreteVarInfo] | None = None,
) -> np.ndarray:
    """Round integer variables to nearest allowed values (or project to nearest range)."""
    x_rounded = x.copy()

    # Round integer variables
    for idx in int_indices:
        if discrete_vars and idx in discrete_vars:
            dvar = discrete_vars[idx]
            # Use the nearest() method which handles both discrete values and ranges
            x_rounded[idx] = dvar.nearest(x[idx])
        else:
            # Standard rounding
            x_rounded[idx] = round(x[idx])

    # Project pure range variables (DiscreteRanges) to nearest valid range
    if discrete_vars:
        for idx, dvar in discrete_vars.items():
            if idx in int_indices:
                continue  # Already handled above
            # Project to nearest value in any range
            x_rounded[idx] = dvar.nearest(x[idx])

    return x_rounded


def create_discrete_child_nodes(
    parent: BBNode,
    branch_idx: int,
    branch_val: float,
    parent_obj: float,
    parent_x: np.ndarray,
    node_counter: int,
    discrete_vars: Dict[int, DiscreteVarInfo] | None = None,
) -> Tuple[BBNode, BBNode]:
    """Create child nodes, respecting discrete allowed values and ranges."""
    current_lb, current_ub = parent.var_bounds.get(branch_idx, (-1e8, 1e8))

    if discrete_vars and branch_idx in discrete_vars:
        dvar = discrete_vars[branch_idx]
        tol = dvar.tolerance

        # Collect all disjuncts (discrete values + ranges) that overlap with current bounds
        # Each disjunct is represented as (lb, ub) - for discrete values, lb == ub
        disjuncts_in_range = []

        # Discrete values
        for v in dvar.allowed_values:
            if current_lb - tol <= v <= current_ub + tol:
                disjuncts_in_range.append((v, v))

        # Ranges (may be truncated by current bounds)
        for r_lb, r_ub in dvar.allowed_ranges:
            # Check if range overlaps with current bounds
            effective_lb = max(r_lb, current_lb)
            effective_ub = min(r_ub, current_ub)
            if effective_lb <= effective_ub + tol:
                disjuncts_in_range.append((effective_lb, effective_ub))

        # Sort by lower bound
        disjuncts_in_range.sort(key=lambda d: d[0])

        if len(disjuncts_in_range) <= 1:
            # Only one disjunct left, should not be branching
            left_bounds = dict(parent.var_bounds)
            right_bounds = dict(parent.var_bounds)
            if disjuncts_in_range:
                d_lb, d_ub = disjuncts_in_range[0]
                left_bounds[branch_idx] = (d_lb, d_ub)
                right_bounds[branch_idx] = (d_lb, d_ub)
            else:
                # No disjuncts - should be infeasible
                left_bounds[branch_idx] = (1e8, -1e8)
                right_bounds[branch_idx] = (1e8, -1e8)
        else:
            # Split disjuncts into two sets based on branch_val
            # Disjuncts entirely below/at branch_val go left
            # Disjuncts entirely above branch_val go right
            # Ranges straddling branch_val are split

            left_disjuncts = []
            right_disjuncts = []

            for d_lb, d_ub in disjuncts_in_range:
                if d_ub <= branch_val:
                    # Entirely below/at branch_val
                    left_disjuncts.append((d_lb, d_ub))
                elif d_lb > branch_val:
                    # Entirely above branch_val
                    right_disjuncts.append((d_lb, d_ub))
                else:
                    # Range straddles branch_val - split it
                    if d_lb <= branch_val:
                        left_disjuncts.append((d_lb, branch_val))
                    if branch_val < d_ub:
                        right_disjuncts.append((branch_val, d_ub))

            # Ensure both sides have something (fallback to splitting in half)
            if not left_disjuncts:
                mid = len(disjuncts_in_range) // 2
                left_disjuncts = disjuncts_in_range[:mid]
                right_disjuncts = disjuncts_in_range[mid:]
            elif not right_disjuncts:
                mid = len(disjuncts_in_range) // 2
                left_disjuncts = disjuncts_in_range[:mid]
                right_disjuncts = disjuncts_in_range[mid:]

            # Compute bounds for each child
            # Left child: from min lower bound to max upper bound of left disjuncts
            # This ensures the NLP solver respects the actual disjunct boundaries
            left_bounds = dict(parent.var_bounds)
            if left_disjuncts:
                left_min = min(d[0] for d in left_disjuncts)
                left_max = max(d[1] for d in left_disjuncts)
                left_bounds[branch_idx] = (left_min, left_max)
            else:
                left_bounds[branch_idx] = (1e8, -1e8)  # Infeasible

            # Right child: from min lower bound to max upper bound of right disjuncts
            right_bounds = dict(parent.var_bounds)
            if right_disjuncts:
                right_min = min(d[0] for d in right_disjuncts)
                right_max = max(d[1] for d in right_disjuncts)
                right_bounds[branch_idx] = (right_min, right_max)
            else:
                right_bounds[branch_idx] = (1e8, -1e8)  # Infeasible
    else:
        # Standard integer branching
        left_bounds = dict(parent.var_bounds)
        left_bounds[branch_idx] = (current_lb, np.floor(branch_val))

        right_bounds = dict(parent.var_bounds)
        right_bounds[branch_idx] = (np.ceil(branch_val), current_ub)

    # Use the actual NLP relaxation objective (parent_obj) as the lower bound
    # This is the correct bound from the parent's NLP solve, not inherited from grandparent
    left_node = BBNode(
        priority=parent_obj,  # Priority for heap ordering (lower = better for best-first)
        node_id=node_counter,
        depth=parent.depth + 1,
        var_bounds=left_bounds,
        parent_solution=parent_x.copy(),
        lower_bound=parent_obj,  # Correct: use parent's NLP objective as child's lower bound
    )

    right_node = BBNode(
        priority=parent_obj,
        node_id=node_counter + 1,
        depth=parent.depth + 1,
        var_bounds=right_bounds,
        parent_solution=parent_x.copy(),
        lower_bound=parent_obj,
    )

    return left_node, right_node


def build_constraints_filtered(
    problem_data: ProblemData, 
    constraints: List
) -> List[Dict]:
    """Build scipy constraint dictionaries from a filtered list of constraints."""
    from ...parser import eval_expression
    from ...compiler import compile_to_function
    
    cons = []
    use_compile = problem_data.compile

    for constraint in constraints:
        # Skip "in" constraints (discrete set membership) - handled by B&B
        if constraint.op == "in":
            continue

        def make_con_fun(c, compile_exprs=use_compile):
            if compile_exprs:
                compiled_left = compile_to_function(c.left)
                compiled_right = compile_to_function(c.right)

                def con_fun(x):
                    var_dict = problem_data.unpack(x)
                    lval = compiled_left(var_dict)
                    rval = compiled_right(var_dict)
                    res = (
                        lval - rval
                        if c.op in [">=", "==", ">>"]
                        else rval - lval
                    )
                    return np.ravel(res)
            else:
                def con_fun(x):
                    var_dict = problem_data.unpack(x)
                    lval = eval_expression(c.left, var_dict)
                    rval = eval_expression(c.right, var_dict)
                    res = (
                        lval - rval
                        if c.op in [">=", "==", ">>"]
                        else rval - lval
                    )
                    return np.ravel(res)

            return con_fun

        con_fun = make_con_fun(constraint)
        con_jac = jacobian(con_fun)
        con_type = "eq" if constraint.op == "==" else "ineq"

        cons.append({
            "type": con_type,
            "fun": con_fun,
            "jac": con_jac,
            "curvature": constraint.curvature,
        })

    return cons


def remove_redundant_equality_constraints(
    cons: List[Dict],
    x0: np.ndarray,
    verbose: bool = False,
    tol: float = 1e-10,
) -> Tuple[List[Dict], int]:
    """
    Remove linearly dependent AFFINE equality constraints.

    Only processes constraints that are affine (constant Jacobian).
    Nonlinear equality constraints are never removed since their
    Jacobian varies with x, making point-wise redundancy checks invalid.

    Uses QR decomposition with column pivoting to identify and remove
    redundant affine constraints. This prevents singular Jacobian errors
    in scipy solvers like SLSQP.

    This is common in graph problems where degree constraints are
    naturally redundant (e.g., sum of in-degrees = sum of out-degrees).

    Args:
        cons: List of scipy constraint dictionaries (with 'curvature' key)
        x0: Initial point for Jacobian evaluation
        verbose: Whether to log information about removed constraints
        tol: Tolerance for determining linear dependence

    Returns:
        Tuple of (filtered constraints, number removed)
    """
    # Separate constraints by type and curvature
    # Only affine equality constraints are candidates for redundancy removal
    affine_eq_cons = []
    other_cons = []

    for i, c in enumerate(cons):
        curvature = c.get("curvature")
        is_affine = curvature in (Curvature.CONSTANT, Curvature.AFFINE)

        if c["type"] == "eq" and is_affine:
            affine_eq_cons.append((i, c))
        else:
            other_cons.append(c)

    if len(affine_eq_cons) <= 1:
        return cons, 0

    # Build Jacobian matrix for affine equality constraints only
    try:
        jac_rows = []
        eq_indices = []
        for orig_idx, c in affine_eq_cons:
            jac = c["jac"](x0)
            if jac.ndim == 1:
                jac = jac.reshape(1, -1)
            for row in jac:
                jac_rows.append(row)
                eq_indices.append(orig_idx)

        if not jac_rows:
            return cons, 0

        from scipy.linalg import qr as scipy_qr

        J = np.vstack(jac_rows)
        n_rows = J.shape[0]

        # Use QR decomposition with column pivoting on J^T to find row rank
        # pivoting=True returns permutation P where P[:rank] are independent rows
        _, R, P = scipy_qr(J.T, pivoting=True)

        # Find rank by counting significant diagonal elements of R
        diag_R = np.abs(np.diag(R))
        max_diag = np.max(diag_R) if len(diag_R) > 0 else 0
        if max_diag > 0:
            rank = int(np.sum(diag_R > tol * max_diag))
        else:
            rank = 0

        if rank == n_rows:
            # All affine equality constraints are independent
            return cons, 0

        # Keep only the first 'rank' rows (those corresponding to
        # linearly independent constraints)
        # P gives the permutation - P[:rank] are the indices of independent rows
        keep_row_indices = set(P[:rank])

        # Map back to original constraint indices
        keep_constraint_indices = set()
        for row_idx in keep_row_indices:
            keep_constraint_indices.add(eq_indices[row_idx])

        # Build filtered constraint list: independent affine eq + all others
        filtered_affine_eq = [c for orig_idx, c in affine_eq_cons if orig_idx in keep_constraint_indices]

        n_removed = len(affine_eq_cons) - len(filtered_affine_eq)
        if n_removed > 0 and verbose:
            logger.info(f"Removed {n_removed} redundant affine equality constraint(s)")

        return filtered_affine_eq + other_cons, n_removed

    except Exception as e:
        logger.debug(f"Constraint redundancy check failed: {e}")
        return cons, 0

