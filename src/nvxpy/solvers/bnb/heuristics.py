"""
Primal Heuristics for Finding Feasible Solutions

This module implements heuristics for finding feasible integer solutions
during branch-and-bound search:

- Simple rounding: Round continuous solution to nearest integers
- Fix-and-optimize: Fix integers and solve NLP for continuous variables
- Feasibility pump: Alternate between NLP and MILP to find feasible points
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Set, Tuple

import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize

from ..base import ProblemData
from .node import DiscreteVarInfo
from .cuts import OACut, generate_oa_cuts

logger = logging.getLogger(__name__)


def run_initial_heuristics(
    problem_data: ProblemData,
    obj_func: Callable,
    cons: List[Dict],
    int_indices: List[int],
    nlp_method: str = "SLSQP",
    nlp_maxiter: int = 1000,
    nlp_ftol: float = 1e-9,
    discrete_vars: Dict[int, DiscreteVarInfo] | None = None,
    fp_max_iterations: int = 0,
    fp_penalty_init: float = 0.1,
    fp_penalty_growth: float = 1.5,
    fp_use_oa: bool = False,
    fp_time_limit: float = 0.5,
) -> Tuple[np.ndarray | None, float]:
    """Run heuristics to find initial feasible solution."""
    from ..scipy_backend import ScipyBackend
    
    int_indices_set = set(int_indices)
    best_x = None
    best_obj = float("inf")

    # 1. Simple rounding of initial point
    x0 = problem_data.x0.copy()
    x_rounded = round_and_fix(
        x0, int_indices, int_indices_set, problem_data, cons,
        nlp_method, nlp_maxiter, nlp_ftol, discrete_vars
    )
    if x_rounded is not None:
        obj_val = obj_func(x_rounded)
        obj = float(obj_val.item()) if hasattr(obj_val, 'item') else float(obj_val)
        if obj < best_obj:
            best_x = x_rounded
            best_obj = obj

    # 2. Solve NLP relaxation then round
    try:
        minimize_kwargs = {
            "method": nlp_method,
            "constraints": cons,
            "options": {"maxiter": nlp_maxiter, "ftol": nlp_ftol},
        }
        if nlp_method in ScipyBackend.GRADIENT_METHODS:
            minimize_kwargs["jac"] = grad(obj_func)
        result = minimize(obj_func, x0, **minimize_kwargs)
        if result.success:
            x_rounded = round_and_fix(
                result.x, int_indices, int_indices_set, problem_data, cons,
                nlp_method, nlp_maxiter, nlp_ftol, discrete_vars
            )
            if x_rounded is not None:
                obj_val = obj_func(x_rounded)
                obj = float(obj_val.item()) if hasattr(obj_val, 'item') else float(obj_val)
                if obj < best_obj:
                    best_x = x_rounded
                    best_obj = obj
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        logger.warning(f"Initial heuristic failed: {e}")

    # 3. Feasibility pump using scipy MILP
    try:
        fp_x, fp_obj = feasibility_pump(
            problem_data, obj_func, cons, int_indices, int_indices_set,
            nlp_method, nlp_maxiter, nlp_ftol, discrete_vars,
            max_iterations=fp_max_iterations,
            penalty_init=fp_penalty_init,
            penalty_growth=fp_penalty_growth,
            use_oa=fp_use_oa,
            time_limit=fp_time_limit,
        )
        if fp_x is not None and fp_obj < best_obj:
            best_x = fp_x
            best_obj = fp_obj
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        logger.debug(f"Feasibility pump failed: {e}")

    return best_x, best_obj


def rounding_heuristic(
    x: np.ndarray,
    int_indices: List[int],
    int_indices_set: Set[int],
    problem_data: ProblemData,
    obj_func: Callable,
    cons: List[Dict],
    nlp_method: str = "SLSQP",
    nlp_maxiter: int = 1000,
    nlp_ftol: float = 1e-9,
    discrete_vars: Dict[int, DiscreteVarInfo] | None = None,
) -> Tuple[np.ndarray | None, float]:
    """Try to round current solution to integer/discrete feasibility."""
    x_rounded = round_and_fix(
        x, int_indices, int_indices_set, problem_data, cons,
        nlp_method, nlp_maxiter, nlp_ftol, discrete_vars
    )
    if x_rounded is not None:
        obj_val = obj_func(x_rounded)
        obj = float(obj_val.item()) if hasattr(obj_val, 'item') else float(obj_val)
        return x_rounded, obj
    return None, float("inf")


def feasibility_pump(
    problem_data: ProblemData,
    obj_func: Callable,
    cons: List[Dict],
    int_indices: List[int],
    int_indices_set: Set[int],
    nlp_method: str = "SLSQP",
    nlp_maxiter: int = 1000,
    nlp_ftol: float = 1e-9,
    discrete_vars: Dict[int, DiscreteVarInfo] | None = None,
    max_iterations: int = 0,
    penalty_init: float = 0.1,
    penalty_growth: float = 1.5,
    use_oa: bool = False,
    time_limit: float = 0.5,
) -> Tuple[np.ndarray | None, float]:
    """
    Feasibility pump heuristic using scipy's MILP solver.

    The feasibility pump alternates between:
    1. Solving a penalized NLP (objective + penalty * distance to integer target)
    2. Rounding to an integer point
    3. Solving a MILP with OA cuts to find a feasible integer projection

    Key improvements over basic FP:
    - OA cuts from NLP solutions are added to the MILP
    - Cycle detection with random perturbation
    - Penalty term in NLP to encourage integer feasibility
    - Proper fix-and-optimize for feasible solutions

    Args:
        problem_data: Problem specification
        obj_func: Objective function
        cons: Scipy constraint dictionaries
        int_indices: Indices of integer variables
        int_indices_set: Set of integer variable indices
        nlp_method: Method for NLP solves
        nlp_maxiter: Max iterations for NLP solves
        nlp_ftol: Tolerance for NLP solves
        discrete_vars: Discrete variable info
        max_iterations: Maximum pump iterations
        penalty_init: Initial penalty weight for distance term
        penalty_growth: Multiplier for penalty each iteration
        use_oa: Whether to add OA cuts to the MILP subproblem
        time_limit: Time limit in seconds for MILP subproblems

    Returns:
        Tuple of (solution, objective) or (None, inf) if no solution found
    """
    from ..scipy_backend import ScipyBackend
    
    # Early exit if disabled
    if max_iterations <= 0:
        return None, float("inf")

    from scipy.optimize import milp, LinearConstraint, Bounds

    n_vars = len(problem_data.x0)
    x = problem_data.x0.copy()

    # Extract simple bounds for MILP
    from .utils import extract_simple_bounds
    simple_bounds = extract_simple_bounds(problem_data)
    lb = np.full(n_vars, -1e8)
    ub = np.full(n_vars, 1e8)
    for idx, (lower, upper) in simple_bounds.items():
        if lower > float("-inf"):
            lb[idx] = lower
        if upper < float("inf"):
            ub[idx] = upper

    # Mark integer variables for MILP
    integrality = np.zeros(n_vars, dtype=int)
    for idx in int_indices:
        integrality[idx] = 1  # 1 = integer variable

    best_x = None
    best_obj = float("inf")

    # Current integer target (updated each iteration)
    z_int = None

    # Cycle detection: track visited integer patterns
    # Limit size to avoid memory issues in large problems
    max_visited_patterns = 1000
    visited_patterns: Set[tuple] = set()

    # OA cuts accumulated during FP iterations (local to FP, not shared with B&B pool)
    # This is intentional: FP is a self-contained heuristic that doesn't seed the main OA pool
    fp_oa_cuts: List[OACut] = []

    # Penalty weight for distance term in NLP (capped to avoid numerical blowup)
    penalty = penalty_init
    max_penalty = 1e6

    for iteration in range(max_iterations):
        # Step 1: Solve NLP relaxation (with optional penalty toward z_int)
        try:
            if z_int is not None and penalty > 0:
                # Add L1 penalty toward current integer target
                # f(x) + penalty * sum(|x_i - z_int_i|) for integer indices
                def penalized_obj(x_val):
                    base_obj = obj_func(x_val)
                    distance = sum(abs(x_val[i] - z_int[i]) for i in int_indices)
                    return base_obj + penalty * distance

                minimize_kwargs = {
                    "method": nlp_method,
                    "constraints": cons,
                    "options": {"maxiter": nlp_maxiter, "ftol": nlp_ftol},
                }
                # Note: gradient of L1 penalty is discontinuous, so we skip jac
                result = minimize(penalized_obj, x, **minimize_kwargs)
            else:
                minimize_kwargs = {
                    "method": nlp_method,
                    "constraints": cons,
                    "options": {"maxiter": nlp_maxiter, "ftol": nlp_ftol},
                }
                if nlp_method in ScipyBackend.GRADIENT_METHODS:
                    minimize_kwargs["jac"] = grad(obj_func)
                result = minimize(obj_func, x, **minimize_kwargs)

            if not result.success:
                break
            x_nlp = result.x
        except Exception:
            break

        # Generate OA cuts at this NLP solution (if enabled)
        if use_oa:
            new_cuts = generate_oa_cuts(x_nlp, cons)
            fp_oa_cuts.extend(new_cuts)
            # Keep cut pool manageable
            if len(fp_oa_cuts) > 100:
                fp_oa_cuts = fp_oa_cuts[-100:]

        # Step 2: Round to integer point
        x_rounded = x_nlp.copy()
        for idx in int_indices:
            if discrete_vars and idx in discrete_vars:
                dvar = discrete_vars[idx]
                x_rounded[idx] = min(
                    dvar.allowed_values, key=lambda v: abs(v - x_nlp[idx])
                )
            else:
                x_rounded[idx] = round(x_nlp[idx])

        # Create pattern for cycle detection
        int_pattern = tuple(x_rounded[i] for i in int_indices)

        # Check for cycle
        if int_pattern in visited_patterns:
            # Cycle detected - apply random perturbation
            logger.debug(f"FP iteration {iteration}: cycle detected, perturbing")
            for idx in int_indices:
                if np.random.random() < 0.3:  # Flip ~30% of integers
                    if discrete_vars and idx in discrete_vars:
                        dvar = discrete_vars[idx]
                        # Pick a random allowed value
                        x_rounded[idx] = np.random.choice(list(dvar.allowed_values))
                    else:
                        # Flip binary or shift integer
                        if lb[idx] == 0 and ub[idx] == 1:
                            x_rounded[idx] = 1.0 - x_rounded[idx]
                        else:
                            x_rounded[idx] += np.random.choice([-1, 1])
                            x_rounded[idx] = np.clip(x_rounded[idx], lb[idx], ub[idx])
            # Update pattern after perturbation
            int_pattern = tuple(x_rounded[i] for i in int_indices)

        # Track pattern for cycle detection (with size limit)
        if len(visited_patterns) < max_visited_patterns:
            visited_patterns.add(int_pattern)
        z_int = x_rounded.copy()  # Update integer target for next iteration

        # Try fix-and-optimize: fix integers and solve for continuous
        fixed_x = round_and_fix(
            x_rounded, int_indices, int_indices_set, problem_data, cons,
            nlp_method, nlp_maxiter, nlp_ftol, discrete_vars
        )

        if fixed_x is not None:
            # Found a feasible solution!
            obj_val = obj_func(fixed_x)
            obj = float(obj_val.item()) if hasattr(obj_val, 'item') else float(obj_val)
            if obj < best_obj:
                best_x = fixed_x.copy()
                best_obj = obj
                logger.debug(f"FP iteration {iteration}: found feasible solution, obj={obj:.4e}")
            # Continue to see if we can find better
            # Increase penalty to stay near this good region (capped)
            penalty = min(penalty * penalty_growth, max_penalty)

        # Step 3: Use MILP to find integer projection with OA cuts
        n_aux = len(int_indices)
        n_total = n_vars + n_aux

        # Objective: minimize L1 distance to NLP solution point
        c = np.zeros(n_total)
        c[n_vars:] = 1.0  # Auxiliary distance terms

        # Bounds
        lb_ext = np.concatenate([lb, np.zeros(n_aux)])
        ub_ext = np.concatenate([ub, np.full(n_aux, 1e8)])

        # Integrality
        integrality_ext = np.concatenate([integrality, np.zeros(n_aux, dtype=int)])

        # Build constraint matrix
        A_rows = []
        b_lower = []
        b_upper = []

        # Distance constraints: |x_i - x_nlp_i| <= d_i
        for aux_idx, orig_idx in enumerate(int_indices):
            # d_i - x_i >= -x_nlp_i
            row1 = np.zeros(n_total)
            row1[orig_idx] = -1.0
            row1[n_vars + aux_idx] = 1.0
            A_rows.append(row1)
            b_lower.append(-x_nlp[orig_idx])
            b_upper.append(np.inf)

            # d_i + x_i >= x_nlp_i
            row2 = np.zeros(n_total)
            row2[orig_idx] = 1.0
            row2[n_vars + aux_idx] = 1.0
            A_rows.append(row2)
            b_lower.append(x_nlp[orig_idx])
            b_upper.append(np.inf)

        # Add OA cuts (linearized constraints)
        for cut in fp_oa_cuts:
            if len(cut.coefficients) == n_vars:
                # Extend coefficients for auxiliary variables (zeros)
                row = np.concatenate([cut.coefficients, np.zeros(n_aux)])
                A_rows.append(row)
                b_lower.append(cut.rhs)
                b_upper.append(np.inf if not cut.is_equality else cut.rhs)

        if A_rows:
            A = np.array(A_rows)
            milp_constraints = LinearConstraint(A, b_lower, b_upper)
        else:
            milp_constraints = None

        try:
            # MILP projection to find integer point closest to NLP solution
            # Note: These MILP solves are not counted in BBStats.lp_solves since
            # FP runs during initial heuristics before B&B stats are tracked
            milp_result = milp(
                c=c,
                constraints=milp_constraints,
                integrality=integrality_ext,
                bounds=Bounds(lb_ext, ub_ext),
                options={"time_limit": time_limit}
            )
            if milp_result.success:
                x = milp_result.x[:n_vars]
            else:
                # MILP failed, use NLP solution with small perturbation
                x = x_nlp.copy()
                for idx in int_indices:
                    x[idx] += np.random.uniform(-0.1, 0.1)
        except Exception:
            # MILP not available, use perturbation
            x = x_nlp.copy()
            for idx in int_indices:
                x[idx] += np.random.uniform(-0.1, 0.1)

        # Increase penalty for next iteration (capped to avoid numerical issues)
        penalty = min(penalty * penalty_growth, max_penalty)

    return best_x, best_obj


def round_and_fix(
    x: np.ndarray,
    int_indices: List[int],
    int_indices_set: Set[int],
    problem_data: ProblemData,
    cons: List[Dict],
    nlp_method: str = "SLSQP",
    nlp_maxiter: int = 1000,
    nlp_ftol: float = 1e-9,
    discrete_vars: Dict[int, DiscreteVarInfo] | None = None,
) -> np.ndarray | None:
    """Round integers/discrete vars and solve NLP for continuous vars."""
    from ..scipy_backend import ScipyBackend
    from ..base import build_objective
    
    x_fixed = x.copy()

    # Round integer/discrete variables
    for idx in int_indices:
        if discrete_vars and idx in discrete_vars:
            # Round to nearest allowed value (handles both discrete values and ranges)
            dvar = discrete_vars[idx]
            x_fixed[idx] = dvar.nearest(x[idx])
        else:
            # Standard rounding
            x_fixed[idx] = round(x_fixed[idx])

    # For pure range variables (DiscreteRanges), we need to:
    # 1. Find which range the current value is closest to
    # 2. Set bounds to that range so NLP stays within it
    range_bounds: Dict[int, Tuple[float, float]] = {}
    if discrete_vars:
        for idx, dvar in discrete_vars.items():
            if idx in int_indices_set:
                continue  # Already handled above
            # Find the range that contains the value, or the nearest range
            val = x[idx]
            best_range = None
            best_dist = float('inf')
            for r_lb, r_ub in dvar.allowed_ranges:
                if r_lb <= val <= r_ub:
                    # Value is in this range
                    best_range = (r_lb, r_ub)
                    break
                else:
                    # Distance to range
                    dist = min(abs(val - r_lb), abs(val - r_ub))
                    if dist < best_dist:
                        best_dist = dist
                        best_range = (r_lb, r_ub)
            if best_range:
                range_bounds[idx] = best_range
                # Also project the value into this range
                x_fixed[idx] = max(best_range[0], min(best_range[1], val))

    # Fix integer variables and constrain range variables
    n = len(x)
    bounds = []
    for i in range(n):
        if i in int_indices_set:
            bounds.append((x_fixed[i], x_fixed[i]))  # Fixed
        elif i in range_bounds:
            bounds.append(range_bounds[i])  # Constrained to one range
        else:
            bounds.append((None, None))

    # Build objective for continuous vars only
    obj_func = build_objective(problem_data)

    try:
        minimize_kwargs = {
            "method": nlp_method,
            "bounds": bounds,
            "constraints": cons,
            "options": {"maxiter": nlp_maxiter, "ftol": nlp_ftol},
        }
        if nlp_method in ScipyBackend.GRADIENT_METHODS:
            minimize_kwargs["jac"] = grad(obj_func)
        result = minimize(obj_func, x_fixed, **minimize_kwargs)
        if result.success:
            # Verify feasibility
            feasible = True
            for con in cons:
                val = con["fun"](result.x)
                if con["type"] == "eq":
                    if np.any(np.abs(val) > 1e-5):
                        feasible = False
                        break
                else:  # ineq
                    if np.any(val < -1e-5):
                        feasible = False
                        break
            if feasible:
                return result.x
    except Exception as e:
        logger.warning(f"Round-and-fix NLP failed: {e}")

    return None

