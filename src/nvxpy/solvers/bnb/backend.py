"""
Branch-and-Bound MINLP Backend

Implements a branch-and-bound algorithm for mixed-integer nonlinear programming
(MINLP) with multiple advanced features for robust solving.

Features:
- Multiple node selection strategies (best-first, depth-first, hybrid)
- Multiple branching strategies (most fractional, pseudocost, strong branching, reliability)
- Outer approximation (OA) cuts for convex constraint tightening (disabled by default)
- Primal heuristics (rounding, feasibility pump using scipy's MILP solver)
- Warm starting from parent solutions
- Node pruning by bound
- Configurable tolerances and limits
- Discrete value constraints (x ^ [1, 10, 100, ...])

Algorithm Details:
- Each node solves an NLP relaxation using scipy.optimize.minimize (default: SLSQP)
- The global lower bound (best_bound) is the minimum NLP relaxation objective
  across all open nodes in the queue
- Pseudocosts are learned from strong branching NLP solves when using
  reliability or strong branching strategies

Outer Approximation (OA) Cuts:
- For CONVEX problems only (set bb_assume_convex=True)
- Only inequality constraints are linearized; equality constraints are skipped.
- Constraint cuts for g(x) >= 0:
  For convex g, linearization gives: g(x*) + ∇g(x*)ᵀ(x - x*) >= 0
  => ∇g(x*)ᵀ x >= ∇g(x*)ᵀ x* - g(x*)
  These cuts are added as linear inequality constraints in node NLPs,
  tightening the feasible region.

WARNING: For non-convex problems, OA cuts can cut off the global optimum!
Only enable OA with bb_assume_convex=True if you are certain the problem is convex.

Note on solver character:
This is a "local" MINLP B&B because each node solves a local NLP (SLSQP) with
no guarantee of finding the global continuous relaxation optimum. For non-convex
problems, the bounds computed are valid lower bounds on local basins, making
this a heuristic global solver. For convex problems with OA constraint cuts,
the method provides valid tightening of the feasible region.
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import Callable, Dict, List, Tuple

import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize

from ...constants import (
    DEFAULT_INT_TOL,
    DEFAULT_ABS_GAP,
    DEFAULT_REL_GAP,
    DEFAULT_NLP_FTOL,
    DEFAULT_CON_TOL,
    DEFAULT_NEAR_ZERO,
)

logger = logging.getLogger(__name__)

from ..base import ProblemData, SolverResult, SolverStats, SolverStatus, extract_simple_bounds
from ..scipy_backend import ScipyBackend

from .node import (
    BBNode,
    BBStats,
    BranchingStrategy,
    NodeSelection,
    PseudocostData,
)
from .cuts import OACut, generate_oa_cuts, prune_cut_pool
from .branching import select_branching_variable
from .heuristics import rounding_heuristic, run_initial_heuristics
from .utils import (
    build_scipy_constraints,
    create_child_nodes,
    get_initial_bounds,
    get_integer_indices,
    get_integer_violations,
    get_scipy_bounds,
    get_warm_start,
    remove_redundant_equality_constraints,
)


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
                - bb_branching: "most_fractional", "pseudocost", "strong", "reliability" (default: "most_fractional")
                - bb_assume_convex: Assume problem is convex (enables OA cuts) (default: False)
                - bb_use_oa_cuts: Enable outer approximation cuts (default: bb_assume_convex)
                  WARNING: OA cuts can cut off optimal solutions for non-convex problems!
                - bb_use_heuristics: Enable primal heuristics (default: True)
                - bb_strong_branch_limit: Max strong branching candidates (default: 5)
                - bb_reliability_limit: Branches before pseudocost reliable (default: 8)
                - bb_max_cuts: Maximum OA cuts to keep in pool (default: 200)
                - bb_cut_max_age: Remove cuts older than this many nodes (default: 50)

                NLP solver options:
                - nlp_method: scipy.optimize.minimize method (default: "SLSQP")
                - nlp_maxiter: Maximum NLP iterations per solve (default: 1000)
                - nlp_ftol: NLP function tolerance (default: 1e-9)

                Feasibility pump options:
                - fp_max_iterations: Maximum FP iterations (default: 0, disabled)
                - fp_penalty_init: Initial penalty weight for distance term (default: 0.1)
                - fp_penalty_growth: Penalty multiplier each iteration (default: 1.5)
                - fp_use_oa: Use OA cuts in FP MILP subproblem (default: False)
                - fp_time_limit: Time limit for MILP subproblems in seconds (default: 0.5)

        Returns:
            SolverResult with the best solution found
        """
        start_time = time.time()
        setup_time = problem_data.setup_time

        # Extract B&B options
        options = dict(solver_options)
        max_nodes = int(options.pop("bb_max_nodes", 10000))
        max_time = float(options.pop("bb_max_time", 300))
        abs_gap = float(options.pop("bb_abs_gap", DEFAULT_ABS_GAP))
        rel_gap = float(options.pop("bb_rel_gap", DEFAULT_REL_GAP))
        int_tol = float(options.pop("bb_int_tol", DEFAULT_INT_TOL))
        # Accept both 'verbose' and 'bb_verbose' for consistency
        verbose = bool(options.pop("bb_verbose", options.pop("verbose", False)))

        # Configure logging for verbose mode
        if verbose:
            logger.setLevel(logging.INFO)
            if not logger.handlers and not logger.parent.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)

        # Strategy options (speed-first defaults)
        node_sel_opt = options.pop("bb_node_selection", "best_first")
        branch_opt = options.pop("bb_branching", "most_fractional")
        assume_convex = bool(options.pop("bb_assume_convex", False))
        use_oa_cuts = bool(options.pop("bb_use_oa_cuts", assume_convex))
        use_heuristics = bool(options.pop("bb_use_heuristics", True))
        strong_limit = int(options.pop("bb_strong_branch_limit", 5))
        reliability_limit = int(options.pop("bb_reliability_limit", 8))
        max_cuts = int(options.pop("bb_max_cuts", 200))
        cut_max_age = int(options.pop("bb_cut_max_age", 50))

        # NLP solver options
        nlp_maxiter = int(options.pop("nlp_maxiter", 1000))
        nlp_ftol = float(options.pop("nlp_ftol", DEFAULT_NLP_FTOL))
        nlp_method = str(options.pop("nlp_method", "SLSQP"))

        # Feasibility pump options
        fp_max_iterations = int(options.pop("fp_max_iterations", 0))
        fp_penalty_init = float(options.pop("fp_penalty_init", 0.1))
        fp_penalty_growth = float(options.pop("fp_penalty_growth", 1.5))
        fp_use_oa = bool(options.pop("fp_use_oa", False))
        fp_time_limit = float(options.pop("fp_time_limit", 0.5))

        # Remaining options are passed directly to the NLP solver
        nlp_options = dict(options)

        # Parse strategy enums
        node_selection = self._parse_enum(
            node_sel_opt, NodeSelection, "bb_node_selection"
        )
        branching = self._parse_enum(branch_opt, BranchingStrategy, "bb_branching")

        if verbose:
            logger.info(f"Problem size: {len(problem_data.x0)} variables")
            logger.info(f"Integer variables: {problem_data.integer_vars}")
            logger.info(f"Total constraints: {len(problem_data.constraints)}")

        # If no integer variables, solve as NLP
        if not problem_data.integer_vars:
            if verbose:
                logger.info("No integer variables - solving as NLP directly")
            return self._solve_nlp(problem_data, nlp_options, nlp_method)

        # Initialize B&B statistics
        stats = BBStats()

        # Get integer variable indices
        int_indices = get_integer_indices(problem_data)
        int_indices_set = set(int_indices)
        n_vars = len(problem_data.x0)

        # Build objective and constraint functions from ProblemData
        obj_func = problem_data.objective_fn
        obj_grad = grad(obj_func)
        cons = build_scipy_constraints(problem_data)

        # Remove redundant equality constraints
        cons, n_removed = remove_redundant_equality_constraints(
            cons, problem_data.x0, verbose
        )

        # Initialize pseudocost data
        pseudocosts: Dict[int, PseudocostData] = {
            idx: PseudocostData() for idx in int_indices
        }

        # OA cuts pool
        oa_cuts: List[OACut] = []

        # Extract simple variable bounds
        simple_bounds = extract_simple_bounds(problem_data)

        # Initialize variable bounds
        initial_var_bounds = get_initial_bounds(simple_bounds)

        # Initialize best solution (incumbent)
        incumbent_x: np.ndarray | None = None
        incumbent_obj = float("inf")

        # Try to find initial solution via heuristics
        if use_heuristics:
            if verbose:
                logger.info("Running initial heuristics...")
            heur_x, heur_obj = run_initial_heuristics(
                problem_data,
                obj_func,
                cons,
                int_indices,
                nlp_method,
                nlp_maxiter,
                nlp_ftol,
                fp_max_iterations,
                fp_penalty_init,
                fp_penalty_growth,
                fp_use_oa,
                fp_time_limit,
            )
            if heur_x is not None:
                incumbent_x = heur_x
                incumbent_obj = float(heur_obj)
                stats.heuristic_solutions += 1
                if verbose:
                    logger.info(
                        f"Heuristic found initial solution: {incumbent_obj:.6e}"
                    )
            elif verbose:
                logger.info("Initial heuristics found no feasible solution")

        # Create root node
        root = BBNode(
            priority=float("-inf"),
            node_id=0,
            depth=0,
            var_bounds=initial_var_bounds,
            parent_solution=problem_data.x0.copy(),
            lower_bound=float("-inf"),
        )

        # Priority queue
        node_queue: List[BBNode] = [root]
        heapq.heapify(node_queue)

        node_counter = 1
        depth_first_counter = 0

        # Track global lower bound (monotonically non-decreasing for minimization)
        # This is the minimum lower bound across all open nodes
        global_lower_bound = float("-inf")

        if verbose:
            logger.info(
                f"Branch-and-Bound: {len(int_indices)} integer variable elements"
            )
            logger.info(
                f"Strategy: {node_selection.value}, Branching: {branching.value}"
            )
            logger.info(f"OA cuts: {use_oa_cuts}, Heuristics: {use_heuristics}")
            logger.info(f"Max nodes: {max_nodes}, Max time: {max_time}s")
            logger.info(
                f"{'Nodes':>8} {'Incumbent':>12} {'Best Bound':>12} {'Gap':>10} {'Time':>8}"
            )
            logger.info("-" * 54)

        # Main B&B loop
        while node_queue:
            # Check termination conditions
            elapsed = time.time() - start_time
            if elapsed > max_time:
                if verbose:
                    logger.info(f"Time limit reached ({max_time}s)")
                break

            if stats.nodes_explored >= max_nodes:
                if verbose:
                    logger.info(f"Node limit reached ({max_nodes})")
                break

            # Update global lower bound from queue
            # The best bound is the minimum lower_bound across all OPEN nodes in the queue
            # IMPORTANT: Use actual minimum, not monotonically non-decreasing, for correctness.
            # A monotone bound can stay artificially high and cause premature termination
            # when child nodes happen to have tighter (lower) relaxation values.
            if node_queue:
                finite_bounds = [
                    n.lower_bound for n in node_queue if np.isfinite(n.lower_bound)
                ]
                if finite_bounds:
                    current_min_bound = min(finite_bounds)
                    # Use actual minimum, not max with previous (allows bound to decrease)
                    global_lower_bound = current_min_bound
            stats.best_bound = global_lower_bound

            # Check gap
            if (
                incumbent_x is not None
                and np.isfinite(stats.best_bound)
                and np.isfinite(incumbent_obj)
            ):
                if abs(incumbent_obj) > DEFAULT_NEAR_ZERO:
                    stats.gap = abs(incumbent_obj - stats.best_bound) / abs(
                        incumbent_obj
                    )
                else:
                    stats.gap = abs(incumbent_obj - stats.best_bound)

                if (
                    stats.gap <= rel_gap
                    or abs(incumbent_obj - stats.best_bound) <= abs_gap
                ):
                    if verbose:
                        logger.info(f"Optimality gap reached (gap={stats.gap:.2e})")
                    break

            # Select node
            node = self._select_node(node_queue, node_selection, depth_first_counter)
            depth_first_counter += 1
            stats.nodes_explored += 1

            # Prune by bound
            if node.lower_bound >= incumbent_obj - abs_gap:
                stats.nodes_pruned += 1
                continue

            # Solve NLP relaxation
            x0 = get_warm_start(node, problem_data)
            bounds = get_scipy_bounds(node, n_vars)

            nlp_result = self._solve_node_nlp(
                obj_func,
                obj_grad,
                x0,
                bounds,
                cons,
                oa_cuts,
                nlp_method,
                nlp_maxiter,
                nlp_ftol,
            )
            stats.nlp_solves += 1

            if nlp_result is None:
                stats.nodes_infeasible += 1
                continue

            x_relaxed, obj_relaxed = nlp_result

            # Update node's bound
            node.lower_bound = obj_relaxed
            node.priority = obj_relaxed

            # Prune by bound
            if obj_relaxed >= incumbent_obj - abs_gap:
                stats.nodes_pruned += 1
                continue

            # Add OA cuts
            if use_oa_cuts:
                new_cuts = generate_oa_cuts(x_relaxed, cons)
                oa_cuts.extend(new_cuts)
                stats.cuts_added += len(new_cuts)

                for cut in oa_cuts:
                    cut.age += 1
                oa_cuts = prune_cut_pool(oa_cuts, max_cuts, cut_max_age)

            # Check integer feasibility
            int_violations = get_integer_violations(x_relaxed, int_indices, int_tol)

            if not int_violations:
                # Integer feasible
                if obj_relaxed < incumbent_obj:
                    incumbent_x = x_relaxed.copy()
                    incumbent_obj = obj_relaxed

                    # Prune nodes with worse bounds
                    node_queue = [
                        n for n in node_queue if n.lower_bound < incumbent_obj - abs_gap
                    ]
                    heapq.heapify(node_queue)
            else:
                # Try heuristics
                if use_heuristics and stats.nodes_explored % 10 == 0:
                    heur_x, heur_obj = rounding_heuristic(
                        x_relaxed,
                        int_indices,
                        int_indices_set,
                        problem_data,
                        obj_func,
                        cons,
                        nlp_method,
                        nlp_maxiter,
                        nlp_ftol,
                    )
                    if heur_x is not None and float(heur_obj) < incumbent_obj:
                        incumbent_x = heur_x
                        incumbent_obj = float(heur_obj)
                        stats.heuristic_solutions += 1

                # Select branching variable
                sb_maxiter = max(50, nlp_maxiter // 10)
                sb_ftol = nlp_ftol * 10
                branch_idx, branch_val = select_branching_variable(
                    int_violations,
                    pseudocosts,
                    branching,
                    x_relaxed,
                    obj_relaxed,
                    obj_func,
                    obj_grad,
                    bounds,
                    cons,
                    strong_limit,
                    reliability_limit,
                    stats,
                    nlp_method,
                    sb_maxiter,
                    sb_ftol,
                )

                # Create child nodes
                left_node, right_node = create_child_nodes(
                    node,
                    branch_idx,
                    branch_val,
                    obj_relaxed,
                    x_relaxed,
                    node_counter,
                )
                node_counter += 2

                heapq.heappush(node_queue, left_node)
                heapq.heappush(node_queue, right_node)

            # Progress output
            if verbose:
                elapsed_now = time.time() - start_time
                bound_str = (
                    f"{stats.best_bound:>12.4e}"
                    if stats.best_bound > -1e30
                    else "        -inf"
                )
                inc_str = (
                    f"{incumbent_obj:>12.4e}"
                    if incumbent_x is not None
                    else "         inf"
                )
                logger.info(
                    f"{stats.nodes_explored:>8} {inc_str} "
                    f"{bound_str} {stats.gap:>10.2e} "
                    f"{elapsed_now:>7.1f}s"
                )

        # Determine final status
        solve_time = time.time() - start_time

        # If queue is empty and we have an incumbent, the tree is exhausted.
        # Tree exhaustion proves optimality ONLY if all nodes were properly
        # handled (pruned by bound or integer-feasible). If nodes were lost
        # due to NLP failures, we cannot claim optimality since those subtrees
        # might contain better solutions.
        if not node_queue and incumbent_x is not None:
            if stats.nodes_infeasible == 0:
                # No NLP failures - tree properly exhausted, incumbent is optimal
                stats.best_bound = incumbent_obj
                stats.gap = 0.0
            # else: some nodes had NLP failures, keep existing gap as uncertain

        if incumbent_x is None:
            status = SolverStatus.INFEASIBLE
            x_sol = problem_data.x0
        elif stats.gap <= rel_gap or (
            stats.best_bound > -1e30
            and abs(incumbent_obj - stats.best_bound) <= abs_gap
        ):
            status = SolverStatus.OPTIMAL
            x_sol = incumbent_x
        else:
            status = SolverStatus.SUBOPTIMAL
            x_sol = incumbent_x

        if verbose:
            logger.info("-" * 54)
            logger.info(f"Status: {status}")
            logger.info(f"Nodes explored: {stats.nodes_explored}")
            logger.info(f"NLP solves: {stats.nlp_solves}")
            if stats.cuts_added > 0:
                logger.info(f"OA cuts added: {stats.cuts_added}")
            logger.info(f"Heuristic solutions: {stats.heuristic_solutions}")
            if stats.nodes_infeasible > 0:
                logger.info(f"Nodes infeasible: {stats.nodes_infeasible}")
            if incumbent_x is not None:
                logger.info(f"Best objective: {incumbent_obj:.6e}")
                if stats.best_bound > -1e30:
                    logger.info(f"Best bound: {stats.best_bound:.6e}")
                    logger.info(f"Gap: {stats.gap:.2e}")

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

    def _solve_nlp(
        self,
        problem_data: ProblemData,
        options: Dict,
        method: str = "SLSQP",
    ) -> SolverResult:
        """Solve a pure NLP (no integer variables)."""
        backend = ScipyBackend()
        return backend.solve(problem_data, method, options)

    def _solve_node_nlp(
        self,
        obj_func: Callable,
        obj_grad: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float | None, float | None]],
        cons: List[Dict],
        oa_cuts: List[OACut],
        method: str = "SLSQP",
        maxiter: int = 1000,
        ftol: float = DEFAULT_NLP_FTOL,
    ) -> Tuple[np.ndarray, float] | None:
        """Solve NLP relaxation at a node."""
        # Add OA cuts as linear constraints
        all_cons = list(cons)

        for cut in oa_cuts:
            if not cut.is_equality:

                def make_cut_fun(c):
                    def cut_fun(x):
                        return np.dot(c.coefficients, x) - c.rhs

                    return cut_fun

                def make_cut_jac(c):
                    def cut_jac(x):
                        return c.coefficients

                    return cut_jac

                all_cons.append(
                    {
                        "type": "ineq",
                        "fun": make_cut_fun(cut),
                        "jac": make_cut_jac(cut),
                    }
                )

        try:
            minimize_kwargs = {
                "method": method,
                "bounds": bounds,
                "constraints": all_cons,
                "options": {"maxiter": maxiter, "ftol": ftol},
            }
            if method in ScipyBackend.GRADIENT_METHODS:
                minimize_kwargs["jac"] = obj_grad
            result = minimize(obj_func, x0, **minimize_kwargs)

            if not result.success:
                return None

            x_sol = result.x

            # Verify constraint feasibility
            con_tol = DEFAULT_CON_TOL
            feasible = True
            max_violation = 0.0
            for con in cons:
                try:
                    con_val = con["fun"](x_sol)
                    if con["type"] == "eq":
                        violation = np.max(np.abs(con_val))
                        if violation > con_tol:
                            feasible = False
                            max_violation = max(max_violation, violation)
                    else:
                        violation = -np.min(con_val)
                        if violation > con_tol:
                            feasible = False
                            max_violation = max(max_violation, violation)
                except Exception as e:
                    logger.warning(f"Constraint evaluation failed: {e}")
                    feasible = False
                    break

            if not feasible:
                logger.debug(
                    f"Node NLP rejected: constraint violation = {max_violation:.2e}"
                )
                return None

            # Track active cuts
            for cut in oa_cuts:
                slack = np.dot(cut.coefficients, x_sol) - cut.rhs
                if abs(slack) < DEFAULT_CON_TOL:
                    cut.times_active += 1

            fun = result.fun
            if np.ndim(fun) > 0:
                fun = np.ravel(fun)[0]
            return x_sol, float(fun)

        except Exception as e:
            logger.warning(f"Node NLP solve failed: {e}")
            return None

    def _parse_enum(self, value, enum_class, option_name: str):
        """Parse an enum value from string or enum instance."""
        if isinstance(value, enum_class):
            return value
        if isinstance(value, str):
            try:
                return enum_class(value)
            except ValueError:
                pass
            value_upper = value.upper()
            for member in enum_class:
                if member.name == value_upper or member.value.upper() == value_upper:
                    return member
            valid = [m.value for m in enum_class]
            raise ValueError(
                f"Invalid value '{value}' for {option_name}. Valid options: {valid}"
            )
        raise ValueError(
            f"Invalid type for {option_name}: expected string or {enum_class.__name__}, "
            f"got {type(value).__name__}"
        )
