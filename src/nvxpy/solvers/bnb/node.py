"""
Branch-and-Bound Node and Statistics Dataclasses

This module contains the core data structures used by the branch-and-bound
MINLP solver, including nodes, statistics, and pseudocost information.

Note: Discrete set constraints (x ^ [values]) are now reformulated to
binary indicator variables during Problem construction. This simplifies
the B&B solver to only handle standard integer branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

import autograd.numpy as np


class NodeSelection(Enum):
    """Node selection strategy."""

    BEST_FIRST = "best_first"  # Always pick node with best bound
    DEPTH_FIRST = "depth_first"  # Pick deepest node (finds feasible solutions faster)
    HYBRID = "hybrid"  # Alternate between best-first and depth-first


class BranchingStrategy(Enum):
    """Variable branching strategy."""

    MOST_FRACTIONAL = "most_fractional"  # Branch on most fractional variable
    PSEUDOCOST = "pseudocost"  # Use historical bound improvements
    STRONG = "strong"  # Solve child NLPs to estimate improvement
    RELIABILITY = "reliability"  # Strong branching until pseudocosts reliable


@dataclass(order=True)
class BBNode:
    """
    A node in the branch-and-bound tree.

    Bound semantics:
    - `lower_bound` is the authoritative bound for this node. For unexplored nodes,
      it is inherited from the parent's NLP relaxation value. After solving the node's
      NLP relaxation, it is updated to the node's own relaxation objective.
    - `priority` mirrors `lower_bound` for heap ordering in best-first selection.
      Both should always be set to the same value when creating nodes.
    - The global best_bound is computed as min(n.lower_bound for n in queue).
    - Pruning uses lower_bound (node.lower_bound >= incumbent - abs_gap).
    """

    # Priority for heap ordering (should mirror lower_bound for best-first)
    priority: float

    # Node data (not used for comparison)
    node_id: int = field(compare=False)
    depth: int = field(compare=False)

    # Bounds on the flattened x vector: index -> (lb, ub)
    var_bounds: Dict[int, Tuple[float, float]] = field(
        compare=False, default_factory=dict
    )

    # Parent solution as warm start
    parent_solution: np.ndarray | None = field(compare=False, default=None)

    # Lower bound: inherited from parent initially, updated to own NLP relaxation after solving
    lower_bound: float = field(compare=False, default=float("-inf"))


@dataclass
class BBStats:
    """Statistics from the branch-and-bound solve."""

    nodes_explored: int = 0
    nodes_pruned: int = 0
    nodes_infeasible: int = 0
    best_bound: float = float("-inf")
    gap: float = float("inf")
    nlp_solves: int = 0
    cuts_added: int = 0
    heuristic_solutions: int = 0
    strong_branch_calls: int = 0


@dataclass
class PseudocostData:
    """Pseudocost information for a variable."""

    down_cost: float = 1.0  # Average obj improvement per unit down
    up_cost: float = 1.0  # Average obj improvement per unit up
    down_count: int = 0  # Number of down branches observed
    up_count: int = 0  # Number of up branches observed
