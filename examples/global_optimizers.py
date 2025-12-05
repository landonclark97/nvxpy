"""
Examples using SciPy global optimizers with NVXPY problems.

Global optimizers are useful for non-convex problems where local optimizers
may get stuck in local minima. NVXPY provides access to several SciPy global
optimization methods:

- differential_evolution: Evolutionary algorithm with constraint support
- dual_annealing: Generalized simulated annealing
- shgo: Simplicial homology global optimization with constraint support
- basinhopping: Basin-hopping with local minimization

Run:
    python examples/global_optimizers.py
"""

import autograd.numpy as np
import nvxpy as nvx


def differential_evolution_constrained():
    """Rastrigin in 2D with a ball constraint, solved by differential_evolution."""
    print("=" * 60)
    print("Differential Evolution (constrained Rastrigin)")
    print("=" * 60)

    x = nvx.Variable((2,), name="x")

    # Rastrigin objective (global minimum at 0 within the feasible ball)
    objective = 20 + nvx.sum(x**2 - 10 * nvx.cos(2 * np.pi * x))

    constraints = [
        x >= -5.12,
        x <= 5.12,
        nvx.sum(x**2) <= 4.0,  # keeps search inside radius 2
    ]

    prob = nvx.Problem(nvx.Minimize(objective), constraints)
    result = prob.solve(
        solver=nvx.DIFF_EVOLUTION,
        solver_options={
            "maxiter": 200,
            "popsize": 20,
            "seed": 0,
        },
        compile=True,
    )

    print(f"  status: {result.status}")
    print(f"  x*: {x.value}")
    print(f"  objective: {objective.value:.6f}")
    print()


def dual_annealing_unconstrained():
    """Multi-well 1D function solved by dual_annealing."""
    print("=" * 60)
    print("Dual Annealing (double-well with oscillations)")
    print("=" * 60)

    x = nvx.Variable(name="x")

    # Double-well with oscillation to show escaping local minima
    objective = (x - 2) ** 2 * (x + 2) ** 2 + nvx.sin(5 * x)

    # dual_annealing requires finite bounds
    prob = nvx.Problem(nvx.Minimize(objective), [x >= -6, x <= 6])
    result = prob.solve(
        solver=nvx.DUAL_ANNEALING,
        solver_options={"seed": 0},
        compile=True,
    )

    print(f"  status: {result.status}")
    print(f"  x*: {np.asarray(x.value).item():.6f}")
    print(f"  objective: {np.asarray(objective.value).item():.6f}")
    print()


def shgo_constrained():
    """SHGO on a constrained 2D problem."""
    print("=" * 60)
    print("SHGO (constrained Rosenbrock)")
    print("=" * 60)

    x = nvx.Variable((2,), name="x")

    # Rosenbrock function: global minimum at (1, 1)
    objective = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    # Constrain to a disk that includes the optimum
    constraints = [
        x >= -2,
        x <= 2,
        nvx.sum(x**2) <= 3,  # disk of radius sqrt(3) ≈ 1.73
    ]

    prob = nvx.Problem(nvx.Minimize(objective), constraints)
    result = prob.solve(
        solver=nvx.SHGO,
        solver_options={"n": 100, "iters": 3},
        compile=True,
    )

    print(f"  status: {result.status}")
    print(f"  x*: {x.value}")
    print(f"  objective: {objective.value:.6f}")
    print("  (optimal is x*=[1,1], f*=0)")
    print()


def basinhopping_unconstrained():
    """Basin-hopping on a multi-modal function without bounds."""
    print("=" * 60)
    print("Basin-hopping (Ackley function, no bounds required)")
    print("=" * 60)

    x = nvx.Variable((2,), name="x")

    # Ackley function: global minimum at (0, 0) with f(0,0) = 0
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq = nvx.sum(x**2)
    sum_cos = nvx.sum(nvx.cos(c * x))
    n = 2
    objective = -a * nvx.exp(-b * nvx.sqrt(sum_sq / n)) - nvx.exp(sum_cos / n) + a + np.e

    prob = nvx.Problem(nvx.Minimize(objective))
    result = prob.solve(
        solver=nvx.BASINHOPPING,
        solver_options={"niter": 50, "seed": 42},
        compile=True,
    )

    print(f"  status: {result.status}")
    print(f"  x*: {x.value}")
    print(f"  objective: {objective.value:.6f}")
    print("  (optimal is x*=[0,0], f*=0)")
    print()


def basinhopping_with_bounds():
    """Basin-hopping with bound constraints extracted from problem."""
    print("=" * 60)
    print("Basin-hopping (with bounds from constraints)")
    print("=" * 60)

    x = nvx.Variable((2,), name="x")

    # Himmelblau's function: has 4 identical local minima
    # f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    objective = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    # Add bounds - basinhopping will use L-BFGS-B when bounds are present
    constraints = [x >= -5, x <= 5]

    prob = nvx.Problem(nvx.Minimize(objective), constraints)
    result = prob.solve(
        solver=nvx.BASINHOPPING,
        solver_options={"niter": 20, "seed": 123},
        compile=True,
    )

    print(f"  status: {result.status}")
    print(f"  x*: {x.value}")
    print(f"  objective: {objective.value:.6f}")
    print("  (4 global minima exist, all with f*=0)")
    print()


def compare_solvers():
    """Compare different global solvers on the same problem."""
    print("=" * 60)
    print("Solver Comparison (Eggholder-like function)")
    print("=" * 60)

    def solve_with(solver_name, solver_enum, **opts):
        x = nvx.Variable((2,), name="x")

        # Eggholder-inspired function with many local minima
        objective = -(x[1] + 47) * nvx.sin(nvx.sqrt(nvx.abs(x[0] / 2 + x[1] + 47))) - x[
            0
        ] * nvx.sin(nvx.sqrt(nvx.abs(x[0] - (x[1] + 47))))

        constraints = [x >= -100, x <= 100]

        prob = nvx.Problem(nvx.Minimize(objective), constraints)
        result = prob.solve(solver=solver_enum, solver_options=opts, compile=True)

        obj_val = np.asarray(objective.value).item()
        return result.status, x.value, obj_val

    # Run each solver
    results = {}

    status, x_opt, f_opt = solve_with(
        "diff_evolution", nvx.DIFF_EVOLUTION, maxiter=100, seed=0
    )
    results["differential_evolution"] = (status, x_opt, f_opt)

    status, x_opt, f_opt = solve_with("dual_annealing", nvx.DUAL_ANNEALING, seed=0)
    results["dual_annealing"] = (status, x_opt, f_opt)

    status, x_opt, f_opt = solve_with("shgo", nvx.SHGO, n=50, iters=2)
    results["shgo"] = (status, x_opt, f_opt)

    # Print comparison
    print(f"  {'Solver':<25} {'Status':<12} {'Objective':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for name, (status, x_opt, f_opt) in results.items():
        print(f"  {name:<25} {status:<12} {f_opt:>12.4f}")
    print()


if __name__ == "__main__":
    differential_evolution_constrained()
    dual_annealing_unconstrained()
    shgo_constrained()
    basinhopping_unconstrained()
    basinhopping_with_bounds()
    compare_solvers()
