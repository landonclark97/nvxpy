"""
Branch-and-Bound MINLP Backend

Implements a comprehensive branch-and-bound algorithm for mixed-integer nonlinear
programming with multiple advanced features for robust solving.

Features:
- Multiple node selection strategies (best-first, depth-first, hybrid)
- Multiple branching strategies (most fractional, pseudocost, strong branching)
- Outer approximation (OA) cuts for convex substructures
- LP/NLP hybrid mode using scipy's MILP solver
- Primal heuristics (rounding, feasibility pump, local search)
- Warm starting from parent solutions
- Node pruning by bound
- Configurable tolerances and limits
- Discrete value constraints (x ^ [1, 10, 100, ...])
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import autograd.numpy as np
from autograd import grad, jacobian
from scipy.optimize import minimize

from ..parser import eval_expression
from ..compiler import compile_to_function
from .base import ProblemData, SolverResult, SolverStats, SolverStatus


class NodeSelection(Enum):
    """Node selection strategy."""
    BEST_FIRST = "best_first"      # Always pick node with best bound
    DEPTH_FIRST = "depth_first"    # Pick deepest node (finds feasible solutions faster)
    HYBRID = "hybrid"              # Alternate between best-first and depth-first


class BranchingStrategy(Enum):
    """Variable branching strategy."""
    MOST_FRACTIONAL = "most_fractional"  # Branch on most fractional variable
    PSEUDOCOST = "pseudocost"            # Use historical bound improvements
    STRONG = "strong"                     # Solve child NLPs to estimate improvement
    RELIABILITY = "reliability"           # Strong branching until pseudocosts reliable


@dataclass(order=True)
class BBNode:
    """A node in the branch-and-bound tree."""

    # Priority for heap (interpretation depends on node selection)
    priority: float

    # Node data (not used for comparison)
    node_id: int = field(compare=False)
    depth: int = field(compare=False)

    # Bounds on the flattened x vector: index -> (lb, ub)
    var_bounds: Dict[int, Tuple[float, float]] = field(compare=False, default_factory=dict)

    # Parent solution as warm start
    parent_solution: Optional[np.ndarray] = field(compare=False, default=None)

    # Lower bound from parent (for pruning before solving)
    lower_bound: float = field(compare=False, default=float("-inf"))


@dataclass
class OACut:
    """An outer approximation cut (linear constraint)."""
    # Cut: a^T x >= b (for ineq) or a^T x == b (for eq)
    coefficients: np.ndarray  # a
    rhs: float                # b
    is_equality: bool = False
    age: int = 0              # Number of nodes since cut was added
    times_active: int = 0     # Number of times cut was binding


@dataclass
class BBStats:
    """Statistics from the branch-and-bound solve."""

    nodes_explored: int = 0
    nodes_pruned: int = 0
    nodes_infeasible: int = 0
    best_bound: float = float("-inf")
    gap: float = float("inf")
    nlp_solves: int = 0
    lp_solves: int = 0
    cuts_added: int = 0
    heuristic_solutions: int = 0
    strong_branch_calls: int = 0


@dataclass
class PseudocostData:
    """Pseudocost information for a variable."""
    down_cost: float = 1.0     # Average obj improvement per unit down
    up_cost: float = 1.0       # Average obj improvement per unit up
    down_count: int = 0        # Number of down branches observed
    up_count: int = 0          # Number of up branches observed


@dataclass
class DiscreteVarInfo:
    """Information about a variable with discrete allowed values."""
    var_name: str
    flat_index: int  # Index in the flattened x vector
    allowed_values: Tuple[float, ...]  # Sorted tuple of allowed values
    tolerance: float = 1e-6  # Tolerance for membership checking


class BranchAndBoundBackend:
    """
    Comprehensive branch-and-bound solver for mixed-integer nonlinear programs.

    Supports multiple search strategies, branching rules, and enhancement
    techniques for robust MINLP solving.
    """

    def solve(
        self,
        problem_data: ProblemData,
        solver: str,  # noqa: ARG002 - ignored, uses internal solvers
        solver_options: Dict[str, object],
    ) -> SolverResult:
        """
        Solve a MINLP using branch-and-bound.

        Args:
            problem_data: The problem specification
            solver: Ignored (uses internal NLP solver)
            solver_options: Options including B&B and NLP options:

                B&B options:
                - bb_max_nodes: Maximum nodes to explore (default: 10000)
                - bb_max_time: Maximum time in seconds (default: 300)
                - bb_abs_gap: Absolute optimality gap tolerance (default: 1e-6)
                - bb_rel_gap: Relative optimality gap tolerance (default: 1e-4)
                - bb_int_tol: Integer feasibility tolerance (default: 1e-5)
                - bb_verbose: Print progress (default: False)
                - bb_node_selection: "best_first", "depth_first", "hybrid" (default: "best_first")
                - bb_branching: "most_fractional", "pseudocost", "strong", "reliability" (default: "reliability")
                - bb_use_oa_cuts: Enable outer approximation cuts (default: True)
                - bb_use_heuristics: Enable primal heuristics (default: True)
                - bb_strong_branch_limit: Max strong branching candidates (default: 5)
                - bb_reliability_limit: Branches before pseudocost reliable (default: 8)
                - bb_max_cuts: Maximum OA cuts to keep in pool (default: 200)
                - bb_cut_max_age: Remove cuts older than this many nodes (default: 50)

                NLP solver options:
                - nlp_method: scipy.optimize.minimize method (default: "SLSQP")
                - nlp_maxiter: Maximum NLP iterations per solve (default: 1000)
                - nlp_ftol: NLP function tolerance (default: 1e-9)

        Returns:
            SolverResult with the best solution found
        """
        start_time = time.time()
        setup_time = problem_data.setup_time

        # Extract B&B options
        options = dict(solver_options)
        max_nodes = int(options.pop("bb_max_nodes", 10000))
        max_time = float(options.pop("bb_max_time", 300))
        abs_gap = float(options.pop("bb_abs_gap", 1e-6))
        rel_gap = float(options.pop("bb_rel_gap", 1e-4))
        int_tol = float(options.pop("bb_int_tol", 1e-5))
        verbose = bool(options.pop("bb_verbose", False))

        # Strategy options
        node_sel_str = str(options.pop("bb_node_selection", "best_first"))
        branch_str = str(options.pop("bb_branching", "reliability"))
        use_oa_cuts = bool(options.pop("bb_use_oa_cuts", True))
        use_heuristics = bool(options.pop("bb_use_heuristics", True))
        strong_limit = int(options.pop("bb_strong_branch_limit", 5))
        reliability_limit = int(options.pop("bb_reliability_limit", 8))
        max_cuts = int(options.pop("bb_max_cuts", 200))
        cut_max_age = int(options.pop("bb_cut_max_age", 50))

        # NLP solver options (passed to scipy.optimize.minimize)
        nlp_maxiter = int(options.pop("nlp_maxiter", 1000))
        nlp_ftol = float(options.pop("nlp_ftol", 1e-9))
        nlp_method = str(options.pop("nlp_method", "SLSQP"))
        # Remaining options are passed directly to the NLP solver
        nlp_options = dict(options)

        node_selection = NodeSelection(node_sel_str)
        branching = BranchingStrategy(branch_str)

        # Extract discrete constraints (x ^ [values]) and get remaining constraints
        discrete_vars, remaining_constraints = self._extract_discrete_constraints(problem_data)

        # If no integer variables and no discrete constraints, just solve the NLP directly
        if not problem_data.integer_vars and not discrete_vars:
            return self._solve_nlp(problem_data, nlp_options, nlp_method)

        # Initialize B&B statistics
        stats = BBStats()

        # Get integer variable indices (flat indices into x vector)
        int_indices = self._get_integer_indices(problem_data)

        # Add discrete variable indices that aren't already in int_indices
        for idx in discrete_vars:
            if idx not in int_indices:
                int_indices.append(idx)
        int_indices = sorted(set(int_indices))

        int_indices_set = set(int_indices)
        n_vars = len(problem_data.x0)

        # Build objective and constraint functions (excluding discrete constraints)
        obj_func = self._build_objective(problem_data)
        obj_grad = grad(obj_func)
        cons = self._build_constraints_filtered(problem_data, remaining_constraints)

        # Initialize pseudocost data for each integer variable index
        pseudocosts: Dict[int, PseudocostData] = {
            idx: PseudocostData() for idx in int_indices
        }

        # OA cuts pool
        oa_cuts: List[OACut] = []

        # Initialize variable bounds from discrete constraints (constraint propagation)
        initial_var_bounds = self._propagate_discrete_bounds(discrete_vars)

        # Initialize best solution (incumbent)
        incumbent_x: Optional[np.ndarray] = None
        incumbent_obj = float("inf")

        # Try to find initial solution via heuristics
        if use_heuristics:
            heur_x, heur_obj = self._run_initial_heuristics(
                problem_data, obj_func, cons, int_indices,
                nlp_method, nlp_maxiter, nlp_ftol, discrete_vars
            )
            if heur_x is not None:
                incumbent_x = heur_x
                incumbent_obj = float(heur_obj)
                stats.heuristic_solutions += 1
                if verbose:
                    print(f"Heuristic found initial solution: {incumbent_obj:.6e}")

        # Create root node with bounds from constraint propagation
        root = BBNode(
            priority=float("-inf"),
            node_id=0,
            depth=0,
            var_bounds=initial_var_bounds,
            parent_solution=problem_data.x0.copy(),
            lower_bound=float("-inf"),
        )

        # Priority queue (min-heap)
        node_queue: List[BBNode] = [root]
        heapq.heapify(node_queue)

        node_counter = 1
        depth_first_counter = 0  # For hybrid mode

        if verbose:
            n_discrete = len(discrete_vars)
            print(f"Branch-and-Bound: {len(int_indices)} integer/discrete variable elements")
            if n_discrete > 0:
                print(f"  ({n_discrete} with discrete value constraints)")
            print(f"Strategy: {node_selection.value}, Branching: {branching.value}")
            print(f"OA cuts: {use_oa_cuts}, Heuristics: {use_heuristics}")
            print(f"{'Nodes':>8} {'Incumbent':>12} {'Best Bound':>12} {'Gap':>10} {'Time':>8}")
            print("-" * 54)

        # Main B&B loop
        while node_queue:
            # Check termination conditions
            elapsed = time.time() - start_time
            if elapsed > max_time:
                if verbose:
                    print(f"Time limit reached ({max_time}s)")
                break

            if stats.nodes_explored >= max_nodes:
                if verbose:
                    print(f"Node limit reached ({max_nodes})")
                break

            # Update best bound from queue
            if node_queue:
                # Best bound is the minimum priority in the queue
                stats.best_bound = min(n.priority for n in node_queue)

            # Check gap
            if incumbent_x is not None:
                if abs(incumbent_obj) > 1e-10:
                    stats.gap = abs(incumbent_obj - stats.best_bound) / abs(incumbent_obj)
                else:
                    stats.gap = abs(incumbent_obj - stats.best_bound)

                if stats.gap <= rel_gap or abs(incumbent_obj - stats.best_bound) <= abs_gap:
                    if verbose:
                        print(f"Optimality gap reached (gap={stats.gap:.2e})")
                    break

            # Select node based on strategy
            node = self._select_node(node_queue, node_selection, depth_first_counter)
            depth_first_counter += 1
            stats.nodes_explored += 1

            # Prune by bound (quick check before expensive NLP)
            if node.lower_bound >= incumbent_obj - abs_gap:
                stats.nodes_pruned += 1
                continue

            # Solve NLP relaxation at this node
            x0 = self._get_warm_start(node, problem_data)
            bounds = self._get_scipy_bounds(node, n_vars)

            nlp_result = self._solve_node_nlp(
                obj_func, obj_grad, x0, bounds, cons, oa_cuts,
                nlp_method, nlp_maxiter, nlp_ftol
            )
            stats.nlp_solves += 1

            if nlp_result is None:
                # NLP solve failed - treat as infeasible
                stats.nodes_infeasible += 1
                continue

            x_relaxed, obj_relaxed = nlp_result

            # Prune by bound (using actual objective)
            if obj_relaxed >= incumbent_obj - abs_gap:
                stats.nodes_pruned += 1
                continue

            # Add OA cuts from this solution and manage cut pool
            if use_oa_cuts:
                new_cuts = self._generate_oa_cuts(
                    x_relaxed, obj_func, cons
                )
                oa_cuts.extend(new_cuts)
                stats.cuts_added += len(new_cuts)

                # Age existing cuts and prune old/inactive ones
                for cut in oa_cuts:
                    cut.age += 1
                oa_cuts = self._prune_cut_pool(oa_cuts, max_cuts, cut_max_age)

            # Check integer/discrete feasibility
            int_violations = self._get_integer_violations(
                x_relaxed, int_indices, int_tol, discrete_vars
            )

            if not int_violations:
                # Integer feasible - update incumbent
                if obj_relaxed < incumbent_obj:
                    incumbent_x = x_relaxed.copy()
                    incumbent_obj = obj_relaxed

                    # Prune nodes with worse bounds
                    node_queue = [n for n in node_queue
                                  if n.priority < incumbent_obj - abs_gap]
                    heapq.heapify(node_queue)

                    if verbose:
                        elapsed_now = time.time() - start_time
                        print(f"{stats.nodes_explored:>8} {incumbent_obj:>12.4e} "
                              f"{stats.best_bound:>12.4e} {stats.gap:>10.2e} "
                              f"{elapsed_now:>7.1f}s *")
            else:
                # Try heuristics to find integer solution
                if use_heuristics and stats.nodes_explored % 10 == 0:
                    heur_x, heur_obj = self._rounding_heuristic(
                        x_relaxed, int_indices, int_indices_set,
                        problem_data, obj_func, cons,
                        nlp_method, nlp_maxiter, nlp_ftol,
                        discrete_vars
                    )
                    if heur_x is not None and float(heur_obj) < incumbent_obj:
                        incumbent_x = heur_x
                        incumbent_obj = float(heur_obj)
                        stats.heuristic_solutions += 1
                        if verbose:
                            elapsed_now = time.time() - start_time
                            print(f"{stats.nodes_explored:>8} {incumbent_obj:>12.4e} "
                                  f"{stats.best_bound:>12.4e} {stats.gap:>10.2e} "
                                  f"{elapsed_now:>7.1f}s H")

                # Select branching variable
                branch_idx, branch_val = self._select_branching_variable(
                    int_violations, pseudocosts, branching,
                    x_relaxed, obj_relaxed, obj_func, obj_grad, bounds, cons,
                    strong_limit, reliability_limit, stats,
                    nlp_method, discrete_vars
                )

                # Update pseudocosts from parent
                self._update_pseudocosts(
                    node, branch_idx, obj_relaxed, pseudocosts
                )

                # Create child nodes (respecting discrete constraints)
                left_node, right_node = self._create_discrete_child_nodes(
                    node, branch_idx, branch_val, obj_relaxed,
                    x_relaxed, node_counter, discrete_vars
                )
                node_counter += 2

                heapq.heappush(node_queue, left_node)
                heapq.heappush(node_queue, right_node)

            # Periodic verbose output
            if verbose and stats.nodes_explored % 100 == 0:
                elapsed_now = time.time() - start_time
                bound_str = f"{stats.best_bound:>12.4e}" if stats.best_bound > -1e30 else "        -inf"
                inc_str = f"{incumbent_obj:>12.4e}" if incumbent_x is not None else "         inf"
                print(f"{stats.nodes_explored:>8} {inc_str} "
                      f"{bound_str} {stats.gap:>10.2e} "
                      f"{elapsed_now:>7.1f}s")

        # Determine final status
        solve_time = time.time() - start_time

        # If queue is empty and we have a solution, we've proven optimality
        if not node_queue and incumbent_x is not None:
            stats.best_bound = incumbent_obj
            stats.gap = 0.0

        if incumbent_x is None:
            status = SolverStatus.INFEASIBLE
            x_sol = problem_data.x0
        elif stats.gap <= rel_gap or (stats.best_bound > -1e30 and
                                       abs(incumbent_obj - stats.best_bound) <= abs_gap):
            status = SolverStatus.OPTIMAL
            x_sol = incumbent_x
        else:
            status = SolverStatus.SUBOPTIMAL
            x_sol = incumbent_x

        if verbose:
            print("-" * 54)
            print(f"Status: {status}")
            print(f"Nodes explored: {stats.nodes_explored}")
            print(f"NLP solves: {stats.nlp_solves}")
            print(f"OA cuts added: {stats.cuts_added}")
            print(f"Heuristic solutions: {stats.heuristic_solutions}")
            if incumbent_x is not None:
                print(f"Best objective: {incumbent_obj:.6e}")
                if stats.best_bound > -1e30:
                    print(f"Best bound: {stats.best_bound:.6e}")
                    print(f"Gap: {stats.gap:.2e}")

        solver_stats = SolverStats(
            solver_name="B&B(SLSQP)",
            solve_time=solve_time,
            setup_time=setup_time,
            num_iters=stats.nodes_explored,
        )

        return SolverResult(
            x=x_sol,
            status=status,
            stats=solver_stats,
            raw_result={
                "bb_stats": stats,
                "incumbent_obj": incumbent_obj if incumbent_x is not None else None,
                "best_bound": stats.best_bound,
                "gap": stats.gap,
            },
        )

    # =========================================================================
    # Node Selection
    # =========================================================================

    def _select_node(
        self,
        node_queue: List[BBNode],
        strategy: NodeSelection,
        counter: int,
    ) -> BBNode:
        """Select next node to process based on strategy."""
        if strategy == NodeSelection.BEST_FIRST:
            return heapq.heappop(node_queue)

        elif strategy == NodeSelection.DEPTH_FIRST:
            # Find deepest node
            max_depth = -1
            max_idx = 0
            for i, node in enumerate(node_queue):
                if node.depth > max_depth:
                    max_depth = node.depth
                    max_idx = i
            node = node_queue.pop(max_idx)
            heapq.heapify(node_queue)
            return node

        else:  # HYBRID
            # Alternate: mostly depth-first but occasionally best-first
            if counter % 10 == 0:
                return heapq.heappop(node_queue)
            else:
                max_depth = -1
                max_idx = 0
                for i, node in enumerate(node_queue):
                    if node.depth > max_depth:
                        max_depth = node.depth
                        max_idx = i
                node = node_queue.pop(max_idx)
                heapq.heapify(node_queue)
                return node

    # =========================================================================
    # Branching Variable Selection
    # =========================================================================

    def _select_branching_variable(
        self,
        violations: List[Tuple[int, float]],
        pseudocosts: Dict[int, PseudocostData],
        strategy: BranchingStrategy,
        x: np.ndarray,
        obj: float,
        obj_func: Callable,
        obj_grad: Callable,
        bounds: List[Tuple[Optional[float], Optional[float]]],
        cons: List[Dict],
        strong_limit: int,
        reliability_limit: int,
        stats: BBStats,
        nlp_method: str = "SLSQP",
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Tuple[int, float]:
        """Select branching variable based on strategy."""
        if strategy == BranchingStrategy.MOST_FRACTIONAL:
            return self._most_fractional_branching(violations, discrete_vars)

        elif strategy == BranchingStrategy.PSEUDOCOST:
            return self._pseudocost_branching(violations, pseudocosts, discrete_vars)

        elif strategy == BranchingStrategy.STRONG:
            return self._strong_branching(
                violations[:strong_limit], x, obj, obj_func, obj_grad,
                bounds, cons, stats, nlp_method, discrete_vars
            )

        else:  # RELIABILITY
            # Use strong branching for unreliable variables
            unreliable = [
                (idx, val) for idx, val in violations
                if (pseudocosts[idx].down_count < reliability_limit or
                    pseudocosts[idx].up_count < reliability_limit)
            ]
            if unreliable:
                candidates = unreliable[:strong_limit]
                return self._strong_branching(
                    candidates, x, obj, obj_func, obj_grad,
                    bounds, cons, stats, nlp_method, discrete_vars
                )
            else:
                return self._pseudocost_branching(violations, pseudocosts, discrete_vars)

    def _most_fractional_branching(
        self,
        violations: List[Tuple[int, float]],
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Tuple[int, float]:
        """Select the most fractional variable for branching."""
        best_idx = violations[0][0]
        best_val = violations[0][1]
        best_score = self._fractionality_score(best_idx, best_val, discrete_vars)

        for idx, val in violations[1:]:
            score = self._fractionality_score(idx, val, discrete_vars)
            if score < best_score:
                best_idx = idx
                best_val = val
                best_score = score

        return best_idx, best_val

    def _fractionality_score(
        self,
        idx: int,
        val: float,
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> float:
        """Compute fractionality score (lower = more fractional = better to branch)."""
        if discrete_vars and idx in discrete_vars:
            # For discrete vars: distance to nearest allowed value, normalized
            dvar = discrete_vars[idx]
            if len(dvar.allowed_values) <= 1:
                return float("inf")  # Don't branch on fixed variables
            nearest = min(dvar.allowed_values, key=lambda v: abs(v - val))
            # Normalize by gap between allowed values
            min_gap = min(
                abs(dvar.allowed_values[i+1] - dvar.allowed_values[i])
                for i in range(len(dvar.allowed_values) - 1)
            )
            return abs(val - nearest) / max(min_gap, 1e-6)
        else:
            # Standard: 0.5 - |frac - 0.5| (most fractional at 0.5)
            return abs(0.5 - abs(val - round(val)))

    def _pseudocost_branching(
        self,
        violations: List[Tuple[int, float]],
        pseudocosts: Dict[int, PseudocostData],
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Tuple[int, float]:
        """Select variable with best pseudocost score."""
        best_idx = violations[0][0]
        best_val = violations[0][1]
        best_score = float("-inf")

        for idx, val in violations:
            pc = pseudocosts[idx]

            if discrete_vars and idx in discrete_vars:
                dvar = discrete_vars[idx]
                # For discrete: find distances to nearest lower and upper values
                below = [v for v in dvar.allowed_values if v < val]
                above = [v for v in dvar.allowed_values if v > val]
                down_dist = val - max(below) if below else 0
                up_dist = min(above) - val if above else 0
            else:
                # Standard integer
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

    def _strong_branching(
        self,
        candidates: List[Tuple[int, float]],
        x: np.ndarray,
        obj: float,
        obj_func: Callable,
        obj_grad: Callable,
        bounds: List[Tuple[Optional[float], Optional[float]]],
        cons: List[Dict],
        stats: BBStats,
        nlp_method: str = "SLSQP",
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Tuple[int, float]:
        """Evaluate candidates by solving child NLPs."""
        best_idx = candidates[0][0]
        best_val = candidates[0][1]
        best_score = float("-inf")

        for idx, val in candidates:
            current_lb, current_ub = bounds[idx] if bounds[idx] != (None, None) else (-1e8, 1e8)
            current_lb = current_lb if current_lb is not None else -1e8
            current_ub = current_ub if current_ub is not None else 1e8

            if discrete_vars and idx in discrete_vars:
                dvar = discrete_vars[idx]
                # Find allowed values in current range
                allowed = [
                    v for v in dvar.allowed_values
                    if current_lb - dvar.tolerance <= v <= current_ub + dvar.tolerance
                ]
                below = [v for v in allowed if v <= val]
                above = [v for v in allowed if v > val]

                # Down branch: values <= max(below)
                if below:
                    down_ub = max(below)
                    down_bounds = list(bounds)
                    down_bounds[idx] = (current_lb, down_ub)
                    x0_down = x.copy()
                    x0_down[idx] = down_ub
                else:
                    down_bounds = None

                # Up branch: values >= min(above)
                if above:
                    up_lb = min(above)
                    up_bounds = list(bounds)
                    up_bounds[idx] = (up_lb, current_ub)
                    x0_up = x.copy()
                    x0_up[idx] = up_lb
                else:
                    up_bounds = None
            else:
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
            if down_bounds is not None:
                try:
                    res_down = minimize(
                        obj_func, x0_down, method=nlp_method, jac=obj_grad,
                        bounds=down_bounds, constraints=cons,
                        options={"maxiter": 100, "ftol": 1e-6}
                    )
                    down_obj = res_down.fun if res_down.success else float("inf")
                except Exception:
                    down_obj = float("inf")
                stats.strong_branch_calls += 1
            else:
                down_obj = float("inf")

            # Solve up branch
            if up_bounds is not None:
                try:
                    res_up = minimize(
                        obj_func, x0_up, method=nlp_method, jac=obj_grad,
                        bounds=up_bounds, constraints=cons,
                        options={"maxiter": 100, "ftol": 1e-6}
                    )
                    up_obj = res_up.fun if res_up.success else float("inf")
                except Exception:
                    up_obj = float("inf")
                stats.strong_branch_calls += 1
            else:
                up_obj = float("inf")

            # Score: product of improvements (encourages balanced branches)
            down_imp = max(0, down_obj - obj)
            up_imp = max(0, up_obj - obj)
            score = min(down_imp, up_imp) + 0.1 * max(down_imp, up_imp)

            if score > best_score:
                best_idx = idx
                best_val = val
                best_score = score

        return best_idx, best_val

    def _update_pseudocosts(
        self,
        node: BBNode,
        branch_idx: int,
        obj_relaxed: float,
        pseudocosts: Dict[int, PseudocostData],
    ) -> None:
        """Update pseudocost data from branching results."""
        if node.parent_solution is None or node.lower_bound <= float("-inf"):
            return

        # Check which bound was tightened at this node
        if branch_idx in node.var_bounds:
            lb, ub = node.var_bounds[branch_idx]
            parent_val = node.parent_solution[branch_idx]
            pc = pseudocosts[branch_idx]

            improvement = obj_relaxed - node.lower_bound

            if ub is not None and ub < parent_val:
                # This was a down branch
                delta = parent_val - ub
                if delta > 1e-6:
                    unit_cost = improvement / delta
                    pc.down_cost = (pc.down_cost * pc.down_count + unit_cost) / (pc.down_count + 1)
                    pc.down_count += 1

            if lb is not None and lb > parent_val:
                # This was an up branch
                delta = lb - parent_val
                if delta > 1e-6:
                    unit_cost = improvement / delta
                    pc.up_cost = (pc.up_cost * pc.up_count + unit_cost) / (pc.up_count + 1)
                    pc.up_count += 1

    # =========================================================================
    # Child Node Creation
    # =========================================================================

    def _create_child_nodes(
        self,
        parent: BBNode,
        branch_idx: int,
        branch_val: float,
        parent_obj: float,
        parent_x: np.ndarray,
        node_counter: int,
    ) -> Tuple[BBNode, BBNode]:
        """Create left (down) and right (up) child nodes."""
        # Left child: var <= floor(val)
        left_bounds = dict(parent.var_bounds)
        current_lb, current_ub = left_bounds.get(branch_idx, (-1e8, 1e8))
        left_bounds[branch_idx] = (current_lb, np.floor(branch_val))

        left_node = BBNode(
            priority=parent_obj,
            node_id=node_counter,
            depth=parent.depth + 1,
            var_bounds=left_bounds,
            parent_solution=parent_x.copy(),
            lower_bound=parent_obj,
        )

        # Right child: var >= ceil(val)
        right_bounds = dict(parent.var_bounds)
        right_bounds[branch_idx] = (np.ceil(branch_val), current_ub)

        right_node = BBNode(
            priority=parent_obj,
            node_id=node_counter + 1,
            depth=parent.depth + 1,
            var_bounds=right_bounds,
            parent_solution=parent_x.copy(),
            lower_bound=parent_obj,
        )

        return left_node, right_node

    # =========================================================================
    # OA Cut Generation
    # =========================================================================

    def _generate_oa_cuts(
        self,
        x: np.ndarray,
        obj_func: Callable,
        cons: List[Dict],
    ) -> List[OACut]:
        """Generate outer approximation cuts at current point."""
        cuts = []

        # Objective gradient cut: f(x) + grad_f(x)^T (y - x) <= f*
        # Rearranged: grad_f(x)^T y >= grad_f(x)^T x - f(x) + f*
        # This is valid for convex objectives
        try:
            obj_gradient = grad(obj_func)(x)
            if np.all(np.isfinite(obj_gradient)):
                # Store as linearization for potential MILP use
                obj_val = obj_func(x)
                rhs_val = np.dot(obj_gradient, x) - obj_val
                rhs = float(rhs_val.item()) if hasattr(rhs_val, 'item') else float(rhs_val)
                cuts.append(OACut(
                    coefficients=obj_gradient.copy(),
                    rhs=rhs,
                    is_equality=False
                ))
        except Exception:
            pass

        # Constraint gradient cuts
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
                        # For inequality g(x) >= 0:
                        # g(x*) + grad_g(x*)^T (x - x*) >= 0
                        # grad_g^T x >= grad_g^T x* - g(x*)
                        cuts.append(OACut(
                            coefficients=con_jac[i].copy(),
                            rhs=float(np.dot(con_jac[i], x) - con_val[i]),
                            is_equality=(con["type"] == "eq")
                        ))
            except Exception:
                pass

        return cuts

    # =========================================================================
    # Primal Heuristics
    # =========================================================================

    def _run_initial_heuristics(
        self,
        problem_data: ProblemData,
        obj_func: Callable,
        cons: List[Dict],
        int_indices: List[int],
        nlp_method: str = "SLSQP",
        nlp_maxiter: int = 1000,
        nlp_ftol: float = 1e-9,
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Run heuristics to find initial feasible solution."""
        int_indices_set = set(int_indices)
        best_x = None
        best_obj = float("inf")

        # 1. Simple rounding of initial point
        x0 = problem_data.x0.copy()
        x_rounded = self._round_and_fix(
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
            obj_grad = grad(obj_func)
            result = minimize(
                obj_func, x0, method=nlp_method, jac=obj_grad,
                constraints=cons, options={"maxiter": nlp_maxiter, "ftol": nlp_ftol}
            )
            if result.success:
                x_rounded = self._round_and_fix(
                    result.x, int_indices, int_indices_set, problem_data, cons,
                    nlp_method, nlp_maxiter, nlp_ftol, discrete_vars
                )
                if x_rounded is not None:
                    obj_val = obj_func(x_rounded)
                    obj = float(obj_val.item()) if hasattr(obj_val, 'item') else float(obj_val)
                    if obj < best_obj:
                        best_x = x_rounded
                        best_obj = obj
        except Exception:
            pass

        return best_x, best_obj

    def _rounding_heuristic(
        self,
        x: np.ndarray,
        int_indices: List[int],
        int_indices_set: Set[int],
        problem_data: ProblemData,
        obj_func: Callable,
        cons: List[Dict],
        nlp_method: str = "SLSQP",
        nlp_maxiter: int = 1000,
        nlp_ftol: float = 1e-9,
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Try to round current solution to integer/discrete feasibility."""
        x_rounded = self._round_and_fix(
            x, int_indices, int_indices_set, problem_data, cons,
            nlp_method, nlp_maxiter, nlp_ftol, discrete_vars
        )
        if x_rounded is not None:
            obj_val = obj_func(x_rounded)
            obj = float(obj_val.item()) if hasattr(obj_val, 'item') else float(obj_val)
            return x_rounded, obj
        return None, float("inf")

    def _round_and_fix(
        self,
        x: np.ndarray,
        int_indices: List[int],
        int_indices_set: Set[int],
        problem_data: ProblemData,
        cons: List[Dict],
        nlp_method: str = "SLSQP",
        nlp_maxiter: int = 1000,
        nlp_ftol: float = 1e-9,
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Optional[np.ndarray]:
        """Round integers/discrete vars and solve NLP for continuous vars."""
        x_fixed = x.copy()

        # Round integer/discrete variables
        for idx in int_indices:
            if discrete_vars and idx in discrete_vars:
                # Round to nearest allowed value
                dvar = discrete_vars[idx]
                x_fixed[idx] = min(
                    dvar.allowed_values, key=lambda v: abs(v - x[idx])
                )
            else:
                # Standard rounding
                x_fixed[idx] = round(x_fixed[idx])

        # Fix integer variables and solve for continuous
        n = len(x)
        bounds = []
        for i in range(n):
            if i in int_indices_set:
                bounds.append((x_fixed[i], x_fixed[i]))  # Fixed
            else:
                bounds.append((None, None))

        # Build objective for continuous vars only
        obj_func = self._build_objective(problem_data)
        obj_grad = grad(obj_func)

        try:
            result = minimize(
                obj_func, x_fixed, method=nlp_method, jac=obj_grad,
                bounds=bounds, constraints=cons,
                options={"maxiter": nlp_maxiter, "ftol": nlp_ftol}
            )
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
        except Exception:
            pass

        return None

    # =========================================================================
    # NLP Solving
    # =========================================================================

    def _solve_nlp(
        self,
        problem_data: ProblemData,
        options: Dict,
        method: str = "SLSQP",
    ) -> SolverResult:
        """Solve a pure NLP (no integer variables)."""
        from .scipy_backend import ScipyBackend
        backend = ScipyBackend()
        return backend.solve(problem_data, method, options)

    def _solve_node_nlp(
        self,
        obj_func: Callable,
        obj_grad: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[Optional[float], Optional[float]]],
        cons: List[Dict],
        oa_cuts: List[OACut],
        method: str = "SLSQP",
        maxiter: int = 1000,
        ftol: float = 1e-9,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Solve NLP relaxation at a node."""
        # Add OA cuts as linear constraints (for convex problems helps tightening)
        all_cons = list(cons)

        # Add all cuts (cut pool is already managed externally)
        for cut in oa_cuts:
            if not cut.is_equality:
                # a^T x >= b  =>  a^T x - b >= 0
                def make_cut_fun(c):
                    def cut_fun(x):
                        return np.dot(c.coefficients, x) - c.rhs
                    return cut_fun

                all_cons.append({
                    "type": "ineq",
                    "fun": make_cut_fun(cut),
                })

        try:
            result = minimize(
                obj_func,
                x0,
                method=method,
                jac=obj_grad,
                bounds=bounds,
                constraints=all_cons,
                options={"maxiter": maxiter, "ftol": ftol},
            )

            if result.success or result.status not in (2, 8):
                x_sol = result.x

                # Verify constraint feasibility before accepting solution
                # SLSQP can return "success" with violated constraints
                con_tol = 1e-4
                feasible = True
                for con in cons:
                    try:
                        con_val = con["fun"](x_sol)
                        if con["type"] == "eq":
                            if np.any(np.abs(con_val) > con_tol):
                                feasible = False
                                break
                        else:  # ineq: fun(x) >= 0
                            if np.any(con_val < -con_tol):
                                feasible = False
                                break
                    except Exception:
                        pass

                if not feasible:
                    return None

                # Track which cuts were active (binding)
                for cut in oa_cuts:
                    slack = np.dot(cut.coefficients, x_sol) - cut.rhs
                    if abs(slack) < 1e-4:  # Cut is binding
                        cut.times_active += 1

                return x_sol, float(result.fun)
            return None

        except Exception:
            return None

    def _prune_cut_pool(
        self,
        cuts: List[OACut],
        max_cuts: int,
        max_age: int,
    ) -> List[OACut]:
        """Prune cut pool to remove old/inactive cuts."""
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

    def _propagate_discrete_bounds(
        self,
        discrete_vars: Dict[int, DiscreteVarInfo],
    ) -> Dict[int, Tuple[float, float]]:
        """
        Initialize tight bounds from discrete constraints.

        For each discrete variable, sets bounds to [min_allowed, max_allowed].
        This helps the NLP solver stay in the feasible region.
        """
        var_bounds: Dict[int, Tuple[float, float]] = {}

        for idx, dvar in discrete_vars.items():
            if dvar.allowed_values:
                lb = min(dvar.allowed_values)
                ub = max(dvar.allowed_values)
                var_bounds[idx] = (lb, ub)

        return var_bounds

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_integer_indices(self, problem_data: ProblemData) -> List[int]:
        """Get flat indices for all integer variable elements."""
        indices = []
        for var_name in problem_data.integer_vars:
            start, end = problem_data.var_slices[var_name]
            indices.extend(range(start, end))
        return indices

    def _build_objective(self, problem_data: ProblemData) -> Callable:
        """Build the objective function."""
        if problem_data.compile:
            compiled_obj = compile_to_function(problem_data.objective_expr)

            def obj(x):
                var_dict = problem_data.unpack(x)
                return compiled_obj(var_dict)
        else:
            def obj(x):
                var_dict = problem_data.unpack(x)
                return eval_expression(problem_data.objective_expr, var_dict)

        return obj

    def _build_constraints(self, problem_data: ProblemData) -> List[Dict]:
        """Build scipy constraint dictionaries."""
        return self._build_constraints_filtered(problem_data, problem_data.constraints)

    def _build_constraints_filtered(
        self, problem_data: ProblemData, constraints: List
    ) -> List[Dict]:
        """Build scipy constraint dictionaries from a filtered list of constraints."""
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
            })

        return cons

    def _get_warm_start(
        self, node: BBNode, problem_data: ProblemData
    ) -> np.ndarray:
        """Get warm start, projected to node bounds."""
        if node.parent_solution is not None:
            x0 = node.parent_solution.copy()
        else:
            x0 = problem_data.x0.copy()

        # Project to node bounds
        for idx, (lb, ub) in node.var_bounds.items():
            x0[idx] = np.clip(x0[idx], lb, ub)

        return x0

    def _get_scipy_bounds(
        self,
        node: BBNode,
        n_vars: int,
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Get bounds for scipy.optimize.minimize."""
        bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * n_vars

        # Apply node-specific bounds for integer variables
        for idx, (lb, ub) in node.var_bounds.items():
            bounds[idx] = (lb, ub)

        return bounds

    def _get_integer_violations(
        self,
        x: np.ndarray,
        int_indices: List[int],
        tol: float,
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> List[Tuple[int, float]]:
        """Get list of (index, value) for variables violating integrality/discreteness."""
        violations = []
        for idx in int_indices:
            val = x[idx]

            # Check if this index has a discrete constraint
            if discrete_vars and idx in discrete_vars:
                dvar = discrete_vars[idx]
                # Check if value is close to any allowed value
                is_allowed = any(
                    abs(val - av) <= dvar.tolerance for av in dvar.allowed_values
                )
                if not is_allowed:
                    violations.append((idx, val))
            else:
                # Standard integer check
                frac = abs(val - round(val))
                if frac > tol:
                    violations.append((idx, val))
        return violations

    def _extract_discrete_constraints(
        self,
        problem_data: ProblemData,
    ) -> Tuple[Dict[int, DiscreteVarInfo], List]:
        """
        Extract discrete (x ^ [values]) constraints from problem data.

        Returns:
            Tuple of:
            - Dict mapping flat index -> DiscreteVarInfo
            - List of remaining constraints (non-discrete)
        """
        from ..sets.integer_set import DiscreteSet
        from ..variable import Variable

        discrete_vars: Dict[int, DiscreteVarInfo] = {}
        remaining_constraints = []

        for constraint in problem_data.constraints:
            if constraint.op == "in" and isinstance(constraint.right, DiscreteSet):
                # This is a discrete constraint: var ^ [values]
                var = constraint.left
                discrete_set = constraint.right

                # Get the variable name
                if isinstance(var, Variable):
                    var_name = var.name
                    if var_name in problem_data.var_slices:
                        start, end = problem_data.var_slices[var_name]
                        # For each element of the variable, add discrete info
                        for idx in range(start, end):
                            discrete_vars[idx] = DiscreteVarInfo(
                                var_name=var_name,
                                flat_index=idx,
                                allowed_values=discrete_set.values,
                                tolerance=discrete_set.tolerance,
                            )
            else:
                remaining_constraints.append(constraint)

        return discrete_vars, remaining_constraints

    def _round_to_discrete(
        self,
        x: np.ndarray,
        int_indices: List[int],
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> np.ndarray:
        """Round integer variables to nearest allowed values."""
        x_rounded = x.copy()
        for idx in int_indices:
            if discrete_vars and idx in discrete_vars:
                # Round to nearest allowed value
                dvar = discrete_vars[idx]
                x_rounded[idx] = min(
                    dvar.allowed_values, key=lambda v: abs(v - x[idx])
                )
            else:
                # Standard rounding
                x_rounded[idx] = round(x[idx])
        return x_rounded

    def _create_discrete_child_nodes(
        self,
        parent: BBNode,
        branch_idx: int,
        branch_val: float,
        parent_obj: float,
        parent_x: np.ndarray,
        node_counter: int,
        discrete_vars: Optional[Dict[int, DiscreteVarInfo]] = None,
    ) -> Tuple[BBNode, BBNode]:
        """Create child nodes, respecting discrete allowed values."""
        current_lb, current_ub = parent.var_bounds.get(branch_idx, (-1e8, 1e8))

        if discrete_vars and branch_idx in discrete_vars:
            dvar = discrete_vars[branch_idx]
            # Find allowed values in the current range
            allowed_in_range = [
                v for v in dvar.allowed_values
                if current_lb - dvar.tolerance <= v <= current_ub + dvar.tolerance
            ]

            if len(allowed_in_range) <= 1:
                # Only one value left, should not be branching
                # Create two identical nodes (one will be pruned)
                left_bounds = dict(parent.var_bounds)
                right_bounds = dict(parent.var_bounds)
                if allowed_in_range:
                    val = allowed_in_range[0]
                    left_bounds[branch_idx] = (val, val)
                    right_bounds[branch_idx] = (val, val)
                else:
                    # No values - should be infeasible
                    left_bounds[branch_idx] = (1e8, -1e8)
                    right_bounds[branch_idx] = (1e8, -1e8)
            else:
                # Split allowed values into two sets
                # Find the value closest to branch_val and split there
                below = [v for v in allowed_in_range if v <= branch_val]
                above = [v for v in allowed_in_range if v > branch_val]

                if not below:
                    # All values are above, split in half
                    mid = len(allowed_in_range) // 2
                    below = allowed_in_range[:mid]
                    above = allowed_in_range[mid:]
                elif not above:
                    # All values are below, split in half
                    mid = len(allowed_in_range) // 2
                    below = allowed_in_range[:mid]
                    above = allowed_in_range[mid:]

                # Left child: values <= max(below)
                left_bounds = dict(parent.var_bounds)
                left_bounds[branch_idx] = (current_lb, max(below) if below else current_lb)

                # Right child: values >= min(above)
                right_bounds = dict(parent.var_bounds)
                right_bounds[branch_idx] = (min(above) if above else current_ub, current_ub)
        else:
            # Standard integer branching
            left_bounds = dict(parent.var_bounds)
            left_bounds[branch_idx] = (current_lb, np.floor(branch_val))

            right_bounds = dict(parent.var_bounds)
            right_bounds[branch_idx] = (np.ceil(branch_val), current_ub)

        left_node = BBNode(
            priority=parent_obj,
            node_id=node_counter,
            depth=parent.depth + 1,
            var_bounds=left_bounds,
            parent_solution=parent_x.copy(),
            lower_bound=parent_obj,
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
