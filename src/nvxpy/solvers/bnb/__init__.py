"""
Branch-and-Bound MINLP Solver

This package implements a comprehensive branch-and-bound algorithm for
mixed-integer nonlinear programming (MINLP).

Modules:
- backend: Main BranchAndBoundBackend solver class
- node: Node, statistics, and variable info dataclasses
- branching: Variable branching strategies
- heuristics: Primal heuristics for finding feasible solutions
- cuts: Outer approximation cut generation and management
- utils: Shared utility functions
"""

from .backend import BranchAndBoundBackend
from .node import (
    BBNode,
    BBStats,
    BranchingStrategy,
    DiscreteVarInfo,
    NodeSelection,
    PseudocostData,
)
from .cuts import OACut, generate_oa_cuts, prune_cut_pool

__all__ = [
    "BranchAndBoundBackend",
    "BBNode",
    "BBStats",
    "BranchingStrategy",
    "DiscreteVarInfo",
    "NodeSelection",
    "PseudocostData",
    "OACut",
    "generate_oa_cuts",
    "prune_cut_pool",
]

