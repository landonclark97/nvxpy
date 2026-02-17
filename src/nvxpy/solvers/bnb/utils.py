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

from ...constants import Curvature, DEFAULT_NEAR_ZERO
from ..base import ProblemData
from .node import BBNode

logger = logging.getLogger(__name__)


def get_initial_bounds(
    simple_bounds: Dict[int, Tuple[float, float]] | None = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Initialize variable bounds from simple variable bounds.

    Note: Discrete set constraints (x ^ [values]) are now reformulated to
    indicator variables during Problem construction, so no special discrete
    bound handling is needed here.
    """
    var_bounds: Dict[int, Tuple[float, float]] = {}

    if simple_bounds:
        var_bounds.update(simple_bounds)

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
) -> List[Tuple[int, float]]:
    """Get list of (index, value) for variables violating integrality.

    Note: Discrete set constraints (x ^ [values]) are now reformulated to
    binary indicator variables during Problem construction. These indicator
    variables are included in int_indices and handled with standard
    integrality checking.
    """
    violations = []

    for idx in int_indices:
        val = x[idx]
        frac = abs(val - round(val))
        if frac > tol:
            violations.append((idx, val))

    return violations


def round_to_integers(
    x: np.ndarray,
    int_indices: List[int],
) -> np.ndarray:
    """Round integer variables to nearest integers.

    Note: Discrete set constraints (x ^ [values]) are now reformulated to
    binary indicator variables. These are handled with standard rounding
    (0 or 1).
    """
    x_rounded = x.copy()

    for idx in int_indices:
        x_rounded[idx] = round(x[idx])

    return x_rounded


def create_child_nodes(
    parent: BBNode,
    branch_idx: int,
    branch_val: float,
    parent_obj: float,
    parent_x: np.ndarray,
    node_counter: int,
) -> Tuple[BBNode, BBNode]:
    """Create child nodes using standard integer branching.

    Note: Discrete set constraints (x ^ [values]) are now reformulated to
    binary indicator variables. These are branched using standard integer
    branching (floor/ceil).
    """
    current_lb, current_ub = parent.var_bounds.get(branch_idx, (-1e8, 1e8))

    # Standard integer branching: left <= floor(val), right >= ceil(val)
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


def build_scipy_constraints(
    problem_data: ProblemData,
) -> List[Dict]:
    """Build scipy constraint dictionaries from problem data.

    Note: Discrete set constraints (x ^ [values]) are reformulated to
    indicator variables during Problem construction. The resulting
    constraints are standard equality/inequality constraints that are
    handled here.
    """
    cons = []
    if problem_data.constraints is None:
        return cons

    for i, constraint in enumerate(problem_data.constraints):
        # "in" constraints should have been reformulated during Problem construction
        if constraint.op == "in":
            raise RuntimeError(
                f"Unexpected 'in' constraint found: {constraint}. "
                "DiscreteSet and DiscreteRanges constraints should be "
                "reformulated to indicator variables during Problem construction."
            )

        # Get the corresponding ConstraintFn
        if i < len(problem_data.constraint_fns):
            cfn = problem_data.constraint_fns[i]
            cons.append(
                {
                    "type": cfn.type,
                    "fun": cfn.fun,
                    "jac": jacobian(cfn.fun),
                    "curvature": getattr(constraint, "curvature", None),
                }
            )

    return cons


def remove_redundant_equality_constraints(
    cons: List[Dict],
    x0: np.ndarray,
    verbose: bool = False,
    tol: float = DEFAULT_NEAR_ZERO,
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
        filtered_affine_eq = [
            c for orig_idx, c in affine_eq_cons if orig_idx in keep_constraint_indices
        ]

        n_removed = len(affine_eq_cons) - len(filtered_affine_eq)
        if n_removed > 0 and verbose:
            logger.info(f"Removed {n_removed} redundant affine equality constraint(s)")

        return filtered_affine_eq + other_cons, n_removed

    except Exception as e:
        logger.debug(f"Constraint redundancy check failed: {e}")
        return cons, 0
