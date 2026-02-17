"""
Branching Variable Selection Strategies

This module implements different strategies for selecting which variable
to branch on in the branch-and-bound tree.

Strategies:
- MOST_FRACTIONAL: Branch on the most fractional variable (simple, fast)
- PSEUDOCOST: Use historical bound improvements to estimate branching impact
- STRONG: Solve child NLPs to estimate actual improvement (expensive but accurate)
- RELIABILITY: Use strong branching until pseudocosts become reliable

Note: Discrete set constraints (x ^ [values]) are now reformulated to
binary indicator variables during Problem construction. These are handled
with standard integer branching strategies.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

import autograd.numpy as np
from scipy.optimize import minimize

from .node import BBStats, BranchingStrategy, PseudocostData
from ...constants import DEFAULT_BRANCH_FTOL

logger = logging.getLogger(__name__)


def select_branching_variable(
    violations: List[Tuple[int, float]],
    pseudocosts: Dict[int, PseudocostData],
    strategy: BranchingStrategy,
    x: np.ndarray,
    obj: float,
    obj_func: Callable,
    obj_grad: Callable,
    bounds: List[Tuple[float | None, float | None]],
    cons: List[Dict],
    strong_limit: int,
    reliability_limit: int,
    stats: BBStats,
    nlp_method: str = "SLSQP",
    nlp_maxiter: int = 100,
    nlp_ftol: float = DEFAULT_BRANCH_FTOL,
) -> Tuple[int, float]:
    """Select branching variable based on strategy."""
    if strategy == BranchingStrategy.MOST_FRACTIONAL:
        return most_fractional_branching(violations)

    elif strategy == BranchingStrategy.PSEUDOCOST:
        return pseudocost_branching(violations, pseudocosts)

    elif strategy == BranchingStrategy.STRONG:
        return strong_branching(
            violations[:strong_limit],
            x,
            obj,
            obj_func,
            obj_grad,
            bounds,
            cons,
            stats,
            nlp_method,
            nlp_maxiter,
            nlp_ftol,
            pseudocosts,
        )

    else:  # RELIABILITY
        # Use strong branching for unreliable variables
        unreliable = [
            (idx, val)
            for idx, val in violations
            if (
                pseudocosts[idx].down_count < reliability_limit
                or pseudocosts[idx].up_count < reliability_limit
            )
        ]
        if unreliable:
            candidates = unreliable[:strong_limit]
            return strong_branching(
                candidates,
                x,
                obj,
                obj_func,
                obj_grad,
                bounds,
                cons,
                stats,
                nlp_method,
                nlp_maxiter,
                nlp_ftol,
                pseudocosts,
            )
        else:
            return pseudocost_branching(violations, pseudocosts)


def most_fractional_branching(
    violations: List[Tuple[int, float]],
) -> Tuple[int, float]:
    """Select the most fractional variable for branching."""
    best_idx = violations[0][0]
    best_val = violations[0][1]
    best_score = _fractionality_score(best_val)

    for idx, val in violations[1:]:
        score = _fractionality_score(val)
        if score < best_score:
            best_idx = idx
            best_val = val
            best_score = score

    return best_idx, best_val


def _fractionality_score(val: float) -> float:
    """Compute fractionality score (lower = more fractional = better to branch).

    Score is 0.5 - |frac - 0.5|, so values closest to 0.5 get the lowest score
    (i.e., most fractional, best to branch on).
    """
    return abs(0.5 - abs(val - round(val)))


def pseudocost_branching(
    violations: List[Tuple[int, float]],
    pseudocosts: Dict[int, PseudocostData],
) -> Tuple[int, float]:
    """Select variable with best pseudocost score."""
    best_idx = violations[0][0]
    best_val = violations[0][1]
    best_score = float("-inf")

    for idx, val in violations:
        pc = pseudocosts[idx]

        # Standard integer branching distances
        down_dist = val - np.floor(val)
        up_dist = np.ceil(val) - val

        down_score = pc.down_cost * down_dist
        up_score = pc.up_cost * up_dist
        score = min(down_score, up_score) + 0.1 * max(down_score, up_score)

        if score > best_score:
            best_idx = idx
            best_val = val
            best_score = score

    return best_idx, best_val


def strong_branching(
    candidates: List[Tuple[int, float]],
    x: np.ndarray,
    obj: float,
    obj_func: Callable,
    obj_grad: Callable,
    bounds: List[Tuple[float | None, float | None]],
    cons: List[Dict],
    stats: BBStats,
    nlp_method: str = "SLSQP",
    nlp_maxiter: int = 100,
    nlp_ftol: float = DEFAULT_BRANCH_FTOL,
    pseudocosts: Dict[int, PseudocostData] | None = None,
) -> Tuple[int, float]:
    """
    Evaluate candidates by solving child NLPs and update pseudocosts.

    Uses reduced iteration limits for strong branching NLPs since we only need
    approximate objective improvements, not fully converged solutions.
    """
    from ..scipy_backend import ScipyBackend

    best_idx = candidates[0][0]
    best_val = candidates[0][1]
    best_score = float("-inf")

    for idx, val in candidates:
        current_lb, current_ub = (
            bounds[idx] if bounds[idx] != (None, None) else (-1e8, 1e8)
        )
        current_lb = current_lb if current_lb is not None else -1e8
        current_ub = current_ub if current_ub is not None else 1e8

        # Standard integer branching
        down_bounds = list(bounds)
        down_bounds[idx] = (current_lb, np.floor(val))
        x0_down = x.copy()
        x0_down[idx] = np.floor(val)

        up_bounds = list(bounds)
        up_bounds[idx] = (np.ceil(val), current_ub)
        x0_up = x.copy()
        x0_up[idx] = np.ceil(val)

        # Solve down branch
        try:
            minimize_kwargs = {
                "method": nlp_method,
                "bounds": down_bounds,
                "constraints": cons,
                "options": {"maxiter": nlp_maxiter, "ftol": nlp_ftol},
            }
            if nlp_method in ScipyBackend.GRADIENT_METHODS:
                minimize_kwargs["jac"] = obj_grad
            res_down = minimize(obj_func, x0_down, **minimize_kwargs)
            down_obj = res_down.fun if res_down.success else float("inf")
        except Exception as e:
            logger.debug(f"Strong branching down-solve failed: {e}")
            down_obj = float("inf")
        stats.strong_branch_calls += 1

        # Solve up branch
        try:
            minimize_kwargs = {
                "method": nlp_method,
                "bounds": up_bounds,
                "constraints": cons,
                "options": {"maxiter": nlp_maxiter, "ftol": nlp_ftol},
            }
            if nlp_method in ScipyBackend.GRADIENT_METHODS:
                minimize_kwargs["jac"] = obj_grad
            res_up = minimize(obj_func, x0_up, **minimize_kwargs)
            up_obj = res_up.fun if res_up.success else float("inf")
        except Exception as e:
            logger.debug(f"Strong branching up-solve failed: {e}")
            up_obj = float("inf")
        stats.strong_branch_calls += 1

        # Score: product of improvements (encourages balanced branches)
        # IMPORTANT: If NLP solve failed (obj = inf), treat as zero improvement.
        # Failed solves provide no information and should NOT be rewarded with
        # infinite scores, which would cause the solver to prefer branching on
        # variables that cause failures.
        if np.isfinite(down_obj):
            down_imp = max(0, down_obj - obj)
        else:
            down_imp = 0.0  # No information from failed solve

        if np.isfinite(up_obj):
            up_imp = max(0, up_obj - obj)
        else:
            up_imp = 0.0  # No information from failed solve

        score = min(down_imp, up_imp) + 0.1 * max(down_imp, up_imp)

        # Update pseudocosts from strong branching results
        if pseudocosts is not None and idx in pseudocosts:
            pc = pseudocosts[idx]

            down_dist = val - np.floor(val)
            up_dist = np.ceil(val) - val

            # Update down pseudocost
            if down_obj < float("inf") and down_dist > 1e-6:
                unit_cost = down_imp / down_dist
                pc.down_cost = (pc.down_cost * pc.down_count + unit_cost) / (
                    pc.down_count + 1
                )
                pc.down_count += 1

            # Update up pseudocost
            if up_obj < float("inf") and up_dist > 1e-6:
                unit_cost = up_imp / up_dist
                pc.up_cost = (pc.up_cost * pc.up_count + unit_cost) / (pc.up_count + 1)
                pc.up_count += 1

        if score > best_score:
            best_idx = idx
            best_val = val
            best_score = score

    return best_idx, best_val
