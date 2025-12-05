"""
Outer Approximation (OA) Cut Generation and Management

This module handles the generation and management of outer approximation cuts
for the branch-and-bound MINLP solver.

OA cuts are used to tighten the NLP relaxation at each node for convex problems.
For convex constraints g(x) >= 0, the linearization is:
    g(x*) + ∇g(x*)ᵀ(x - x*) >= 0
    => ∇g(x*)ᵀ x >= ∇g(x*)ᵀ x* - g(x*)

WARNING: For non-convex problems, OA cuts can cut off the global optimum!
Only use OA cuts if you are certain the problem is convex.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import autograd.numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OACut:
    """
    An outer approximation cut (linear constraint).

    For constraint cuts: a^T x >= b (linearization of g(x) >= 0)

    For convex constraints g(x) >= 0, the linearization is:
        g(x*) + ∇g(x*)ᵀ(x - x*) >= 0
        => ∇g(x*)ᵀ x >= ∇g(x*)ᵀ x* - g(x*)

    Note: Epigraph cuts for the objective (t >= f(x)) are NOT generated here.
    They would require an explicit epigraph variable t in the formulation.
    The NLP relaxations minimize f(x) directly without epigraph reformulation.
    """
    coefficients: np.ndarray  # a (gradient coefficients for x)
    rhs: float                # b (right-hand side)
    is_equality: bool = False
    age: int = 0              # Number of nodes since cut was added
    times_active: int = 0     # Number of times cut was binding


def generate_oa_cuts(x: np.ndarray, cons: List[Dict]) -> List[OACut]:
    """
    Generate outer approximation cuts at current point.

    For convex constraints g(x) >= 0, the linearization is:
        g(x*) + ∇g(x*)ᵀ(x - x*) >= 0
        => ∇g(x*)ᵀ x >= ∇g(x*)ᵀ x* - g(x*)

    Implementation notes:
    - Only inequality constraints are used as OA cuts in _solve_node_nlp.
      Equality constraint linearizations are generated but filtered out
      when adding cuts to the NLP (is_equality=True cuts are skipped).
    - Epigraph cuts for the objective would require an explicit epigraph
      variable t in the formulation (min t s.t. t >= f(x)). Since our NLP
      relaxations minimize f(x) directly, epigraph cuts are not applicable.

    Args:
        x: Current solution point
        cons: List of scipy constraint dictionaries

    Returns:
        List of OACut objects (inequality cuts are used, equality cuts stored
        but not applied in node NLPs)
    """
    cuts = []

    for con in cons:
        try:
            con_val = con["fun"](x)
            con_jac = con["jac"](x)

            if con_val.ndim == 0:
                con_val = np.array([con_val])
            if con_jac.ndim == 1:
                con_jac = con_jac.reshape(1, -1)

            for i in range(len(con_val)):
                if np.all(np.isfinite(con_jac[i])):
                    grad_dot_x = np.dot(con_jac[i], x)
                    rhs = float(grad_dot_x - con_val[i])
                    cuts.append(OACut(
                        coefficients=con_jac[i].copy(),
                        rhs=rhs,
                        is_equality=(con["type"] == "eq"),
                    ))
        except Exception as e:
            logger.debug(f"Constraint cut generation failed: {e}")

    return cuts


def prune_cut_pool(
    cuts: List[OACut],
    max_cuts: int,
    max_age: int,
) -> List[OACut]:
    """
    Prune cut pool to remove old/inactive cuts.
    
    Args:
        cuts: List of OA cuts
        max_cuts: Maximum number of cuts to keep
        max_age: Maximum age before considering removal
        
    Returns:
        Filtered list of cuts
    """
    if len(cuts) <= max_cuts:
        # Just remove very old inactive cuts
        return [c for c in cuts if c.age < max_age or c.times_active > 0]

    # Score cuts: newer and more active cuts are better
    # Keep cuts that are either recent or frequently active
    def cut_score(cut: OACut) -> float:
        # Higher score = better cut to keep
        recency = max(0, max_age - cut.age) / max_age
        activity = min(cut.times_active, 10) / 10  # Cap activity score
        return recency * 0.4 + activity * 0.6

    scored = [(cut_score(c), i, c) for i, c in enumerate(cuts)]
    scored.sort(reverse=True)

    return [c for _, _, c in scored[:max_cuts]]

