"""
Branch-and-Bound MINLP Solver

This package implements a comprehensive branch-and-bound algorithm for
mixed-integer nonlinear programming (MINLP).

Modules:
- backend: Main BranchAndBoundBackend solver class
- node: Node, statistics, and pseudocost dataclasses
- branching: Variable branching strategies
- heuristics: Primal heuristics for finding feasible solutions
- cuts: Outer approximation cut generation and management
- utils: Shared utility functions

Note: Discrete set constraints (x ^ [values]) are now reformulated to
binary indicator variables during Problem construction, so the B&B solver
only handles standard integer branching.
"""

from .backend import BranchAndBoundBackend
from .node import (
    BBNode,
    BBStats,
    BranchingStrategy,
    NodeSelection,
    PseudocostData,
)
from .cuts import OACut, generate_oa_cuts, prune_cut_pool

__all__ = [
    "BranchAndBoundBackend",
    "BBNode",
    "BBStats",
    "BranchingStrategy",
    "NodeSelection",
    "PseudocostData",
    "OACut",
    "generate_oa_cuts",
    "prune_cut_pool",
]
