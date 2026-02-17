"""
Benchmark: Compiled nvxpy expressions vs. raw Python/NumPy functions

This test compares the evaluation speed of:
1. Raw Python/NumPy functions (the baseline - what you'd write by hand)
2. nvxpy interpreted evaluation (original eval_expression)
3. nvxpy compiled evaluation (compile_to_function)

The goal is to measure the overhead of the nvxpy DSL compared to hand-written code.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict

import sys

sys.path.insert(0, "src")

import nvxpy as nvx
from nvxpy.parser import eval_expression
from nvxpy.compiler import compile_to_function
from nvxpy.variable import Variable


def reset_variable_ids():
    """Reset variable ID counter between tests."""
    Variable._ids = 0
    Variable._used_names = set()


@dataclass
class ComparisonResult:
    """Result of comparing Python vs nvxpy evaluation."""

    name: str
    python_time: float
    interp_time: float
    compiled_time: float
    interp_overhead: float  # interp_time / python_time
    compiled_overhead: float  # compiled_time / python_time
    results_match: bool

    def __str__(self):
        return (
            f"{self.name:45s} | "
            f"python: {self.python_time * 1000:7.3f}ms | "
            f"interp: {self.interp_time * 1000:7.3f}ms ({self.interp_overhead:5.2f}x) | "
            f"compiled: {self.compiled_time * 1000:7.3f}ms ({self.compiled_overhead:5.2f}x) | "
            f"match: {self.results_match}"
        )


def benchmark_comparison(
    name: str,
    python_func: Callable,
    nvxpy_expr,
    var_dict: Dict[str, np.ndarray],
    num_evals: int = 1000,
) -> ComparisonResult:
    """
    Compare Python function vs nvxpy expression evaluation.
    """
    # Get reference result from Python
    python_result = python_func(var_dict)

    # Time Python function
    start = time.perf_counter()
    for _ in range(num_evals):
        _ = python_func(var_dict)
    python_time = (time.perf_counter() - start) / num_evals

    # Time interpreted nvxpy
    start = time.perf_counter()
    for _ in range(num_evals):
        interp_result = eval_expression(nvxpy_expr, var_dict)
    interp_time = (time.perf_counter() - start) / num_evals

    # Compile and time compiled nvxpy
    compiled_func = compile_to_function(nvxpy_expr)
    start = time.perf_counter()
    for _ in range(num_evals):
        compiled_result = compiled_func(var_dict)
    compiled_time = (time.perf_counter() - start) / num_evals

    # Check results match
    results_match = np.allclose(
        python_result, interp_result, rtol=1e-10, atol=1e-10
    ) and np.allclose(python_result, compiled_result, rtol=1e-10, atol=1e-10)

    return ComparisonResult(
        name=name,
        python_time=python_time,
        interp_time=interp_time,
        compiled_time=compiled_time,
        interp_overhead=interp_time / python_time if python_time > 0 else float("inf"),
        compiled_overhead=compiled_time / python_time
        if python_time > 0
        else float("inf"),
        results_match=results_match,
    )


# =============================================================================
# BENCHMARK SUITE - Simple to Complex
# =============================================================================


def bench_simple_add():
    """Simple: x + y"""
    reset_variable_ids()

    # Python version
    def python_func(v):
        return v["x"] + v["y"]

    # nvxpy version
    x = nvx.Variable((100,), name="x")
    y = nvx.Variable((100,), name="y")
    expr = x + y

    var_dict = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
    }

    return benchmark_comparison("simple: x + y", python_func, expr, var_dict)


def bench_simple_arithmetic():
    """Simple: x + y * 2 - z / 3"""
    reset_variable_ids()

    def python_func(v):
        return v["x"] + v["y"] * 2 - v["z"] / 3

    x = nvx.Variable((100,), name="x")
    y = nvx.Variable((100,), name="y")
    z = nvx.Variable((100,), name="z")
    expr = x + y * 2 - z / 3

    var_dict = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
        "z": np.random.randn(100),
    }

    return benchmark_comparison("simple: x + y*2 - z/3", python_func, expr, var_dict)


def bench_norm_squared():
    """Medium: ||x - y||^2"""
    reset_variable_ids()

    def python_func(v):
        return np.linalg.norm(v["x"] - v["y"]) ** 2

    x = nvx.Variable((100,), name="x")
    y = nvx.Variable((100,), name="y")
    expr = nvx.norm(x - y) ** 2

    var_dict = {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
    }

    return benchmark_comparison("medium: ||x - y||^2", python_func, expr, var_dict)


def bench_matrix_multiply():
    """Medium: A @ B"""
    reset_variable_ids()

    def python_func(v):
        return v["A"] @ v["B"]

    A = nvx.Variable((50, 50), name="A")
    B = nvx.Variable((50, 50), name="B")
    expr = A @ B

    var_dict = {
        "A": np.random.randn(50, 50),
        "B": np.random.randn(50, 50),
    }

    return benchmark_comparison(
        "medium: A @ B", python_func, expr, var_dict, num_evals=500
    )


def bench_matrix_expr():
    """Medium: A @ B + B.T @ A - A * B"""
    reset_variable_ids()

    def python_func(v):
        A, B = v["A"], v["B"]
        return A @ B + B.T @ A - A * B

    A = nvx.Variable((30, 30), name="A")
    B = nvx.Variable((30, 30), name="B")
    expr = A @ B + B.T @ A - A * B

    var_dict = {
        "A": np.random.randn(30, 30),
        "B": np.random.randn(30, 30),
    }

    return benchmark_comparison(
        "medium: A@B + B.T@A - A*B", python_func, expr, var_dict, num_evals=500
    )


def bench_trace_and_norm():
    """Medium: trace(A @ B.T) + ||A - B||_F^2"""
    reset_variable_ids()

    def python_func(v):
        A, B = v["A"], v["B"]
        return np.trace(A @ B.T) + np.linalg.norm(A - B, ord="fro") ** 2

    A = nvx.Variable((30, 30), name="A")
    B = nvx.Variable((30, 30), name="B")
    expr = nvx.trace(A @ B.T) + nvx.norm(A - B, ord="fro") ** 2

    var_dict = {
        "A": np.random.randn(30, 30),
        "B": np.random.randn(30, 30),
    }

    return benchmark_comparison(
        "medium: trace(A@B.T) + ||A-B||_F^2", python_func, expr, var_dict, num_evals=500
    )


def bench_sum_of_norms():
    """Complex: sum of squared norms over multiple vectors"""
    reset_variable_ids()

    n_vecs = 10

    def python_func(v):
        total = 0.0
        for i in range(n_vecs):
            total += np.linalg.norm(v[f"x{i}"]) ** 2
        return total

    vars = [nvx.Variable((50,), name=f"x{i}") for i in range(n_vecs)]
    expr = nvx.norm(vars[0]) ** 2
    for i in range(1, n_vecs):
        expr = expr + nvx.norm(vars[i]) ** 2

    var_dict = {f"x{i}": np.random.randn(50) for i in range(n_vecs)}

    return benchmark_comparison(
        f"complex: sum of {n_vecs} squared norms",
        python_func,
        expr,
        var_dict,
        num_evals=500,
    )


def bench_pairwise_distances():
    """Complex: sum of pairwise squared distances"""
    reset_variable_ids()

    n_vecs = 5

    def python_func(v):
        total = 0.0
        for i in range(n_vecs):
            for j in range(i + 1, n_vecs):
                total += np.linalg.norm(v[f"x{i}"] - v[f"x{j}"]) ** 2
        return total

    vars = [nvx.Variable((30,), name=f"x{i}") for i in range(n_vecs)]

    # Build pairwise distance expression
    pairs = []
    for i in range(n_vecs):
        for j in range(i + 1, n_vecs):
            pairs.append(nvx.norm(vars[i] - vars[j]) ** 2)

    expr = pairs[0]
    for p in pairs[1:]:
        expr = expr + p

    var_dict = {f"x{i}": np.random.randn(30) for i in range(n_vecs)}

    return benchmark_comparison(
        "complex: pairwise distances (5 vecs)",
        python_func,
        expr,
        var_dict,
        num_evals=500,
    )


def bench_polynomial():
    """Complex: polynomial x + x^2 + x^3 + ... + x^10"""
    reset_variable_ids()

    def python_func(v):
        x = v["x"]
        result = x.copy()
        for i in range(2, 11):
            result = result + x**i
        return result

    x = nvx.Variable((50,), name="x")
    expr = x
    for i in range(2, 11):
        expr = expr + x**i

    var_dict = {"x": np.random.randn(50) * 0.5}  # Small values to avoid overflow

    return benchmark_comparison(
        "complex: polynomial degree 10", python_func, expr, var_dict, num_evals=500
    )


def bench_matrix_chain():
    """Complex: A @ B @ C @ D"""
    reset_variable_ids()

    def python_func(v):
        return v["A"] @ v["B"] @ v["C"] @ v["D"]

    A = nvx.Variable((20, 20), name="A")
    B = nvx.Variable((20, 20), name="B")
    C = nvx.Variable((20, 20), name="C")
    D = nvx.Variable((20, 20), name="D")
    expr = A @ B @ C @ D

    var_dict = {
        "A": np.random.randn(20, 20) * 0.3,
        "B": np.random.randn(20, 20) * 0.3,
        "C": np.random.randn(20, 20) * 0.3,
        "D": np.random.randn(20, 20) * 0.3,
    }

    return benchmark_comparison(
        "complex: A @ B @ C @ D", python_func, expr, var_dict, num_evals=500
    )


def bench_least_squares_objective():
    """Complex: ||A @ x - b||^2 + lambda * ||x||^2 (ridge regression objective)"""
    reset_variable_ids()

    lam = 0.1

    def python_func(v):
        A, x, b = v["A"], v["x"], v["b"]
        return np.linalg.norm(A @ x - b) ** 2 + lam * np.linalg.norm(x) ** 2

    A = nvx.Variable((50, 20), name="A")
    x = nvx.Variable((20,), name="x")
    b = nvx.Variable((50,), name="b")
    expr = nvx.norm(A @ x - b) ** 2 + lam * nvx.norm(x) ** 2

    var_dict = {
        "A": np.random.randn(50, 20),
        "x": np.random.randn(20),
        "b": np.random.randn(50),
    }

    return benchmark_comparison(
        "complex: ridge regression objective",
        python_func,
        expr,
        var_dict,
        num_evals=500,
    )


def bench_matrix_factorization_objective():
    """Complex: ||X - U @ V.T||_F^2 + lambda * (||U||_F^2 + ||V||_F^2)"""
    reset_variable_ids()

    lam = 0.01
    X_const = np.random.randn(30, 20)

    def python_func(v):
        U, V = v["U"], v["V"]
        return np.linalg.norm(X_const - U @ V.T, ord="fro") ** 2 + lam * (
            np.linalg.norm(U, ord="fro") ** 2 + np.linalg.norm(V, ord="fro") ** 2
        )

    U = nvx.Variable((30, 5), name="U")
    V = nvx.Variable((20, 5), name="V")
    expr = nvx.norm(X_const - U @ V.T, ord="fro") ** 2 + lam * (
        nvx.norm(U, ord="fro") ** 2 + nvx.norm(V, ord="fro") ** 2
    )

    var_dict = {
        "U": np.random.randn(30, 5),
        "V": np.random.randn(20, 5),
    }

    return benchmark_comparison(
        "complex: matrix factorization obj", python_func, expr, var_dict, num_evals=500
    )


def bench_deep_expression():
    """Complex: deeply nested expression"""
    reset_variable_ids()

    def python_func(v):
        x, y, z = v["x"], v["y"], v["z"]
        t1 = x + y
        t2 = t1 * z
        t3 = t2 - x
        t4 = t3 / (y + 1)
        t5 = t4**2
        t6 = np.sqrt(np.abs(t5) + 1)
        t7 = t6 + t1
        t8 = t7 * t2
        return np.sum(t8)

    x = nvx.Variable((50,), name="x")
    y = nvx.Variable((50,), name="y")
    z = nvx.Variable((50,), name="z")

    t1 = x + y
    t2 = t1 * z
    t3 = t2 - x
    t4 = t3 / (y + 1)
    t5 = t4**2
    # Note: nvxpy doesn't have sqrt, so we use ** 0.5
    t6 = (nvx.abs(t5) + 1) ** 0.5
    t7 = t6 + t1
    t8 = t7 * t2
    expr = nvx.sum(t8)

    var_dict = {
        "x": np.random.randn(50) + 2,  # Ensure positive for sqrt
        "y": np.random.randn(50) + 2,
        "z": np.random.randn(50),
    }

    return benchmark_comparison(
        "complex: deep nested expression", python_func, expr, var_dict, num_evals=500
    )


def bench_indexing_heavy():
    """Complex: lots of indexing operations"""
    reset_variable_ids()

    def python_func(v):
        A = v["A"]
        return (
            A[0, :]
            + A[1, :]
            + A[2, :]
            + A[:, 0]
            + A[:, 1]
            + A[:, 2]
            + np.sum(A[3:7, 3:7])
        )

    A = nvx.Variable((20, 20), name="A")
    expr = (
        A[0, :] + A[1, :] + A[2, :] + A[:, 0] + A[:, 1] + A[:, 2] + nvx.sum(A[3:7, 3:7])
    )

    var_dict = {"A": np.random.randn(20, 20)}

    return benchmark_comparison(
        "complex: indexing heavy", python_func, expr, var_dict, num_evals=500
    )


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================


def run_all_benchmarks():
    """Run all comparison benchmarks."""
    benchmarks = [
        # Simple
        bench_simple_add,
        bench_simple_arithmetic,
        # Medium
        bench_norm_squared,
        bench_matrix_multiply,
        bench_matrix_expr,
        bench_trace_and_norm,
        # Complex
        bench_sum_of_norms,
        bench_pairwise_distances,
        bench_polynomial,
        bench_matrix_chain,
        bench_least_squares_objective,
        bench_matrix_factorization_objective,
        bench_deep_expression,
        bench_indexing_heavy,
    ]

    print("=" * 140)
    print("Benchmark: nvxpy compiled expressions vs. raw Python/NumPy")
    print("=" * 140)
    print()
    print("Overhead = nvxpy_time / python_time (1.0x means same speed as Python)")
    print()

    results = []
    for bench_fn in benchmarks:
        try:
            result = bench_fn()
            results.append(result)
            print(result)
        except Exception as e:
            print(f"FAILED: {bench_fn.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print()
    print("=" * 140)
    print("SUMMARY")
    print("=" * 140)

    # Calculate aggregates
    avg_interp_overhead = sum(r.interp_overhead for r in results) / len(results)
    avg_compiled_overhead = sum(r.compiled_overhead for r in results) / len(results)
    all_match = all(r.results_match for r in results)

    print(f"Average interpreter overhead vs Python:  {avg_interp_overhead:.2f}x")
    print(f"Average compiled overhead vs Python:     {avg_compiled_overhead:.2f}x")
    print(f"All results match:                       {all_match}")
    print()

    # Breakdown by complexity
    simple = results[:2]
    medium = results[2:6]
    complex_results = results[6:]

    print("By complexity (compiled overhead vs Python):")
    if simple:
        avg = sum(r.compiled_overhead for r in simple) / len(simple)
        print(f"  Simple expressions:  {avg:.2f}x overhead")
    if medium:
        avg = sum(r.compiled_overhead for r in medium) / len(medium)
        print(f"  Medium expressions:  {avg:.2f}x overhead")
    if complex_results:
        avg = sum(r.compiled_overhead for r in complex_results) / len(complex_results)
        print(f"  Complex expressions: {avg:.2f}x overhead")

    print()
    print("Interpretation:")
    print("  - Overhead < 2x: Excellent - minimal DSL cost")
    print("  - Overhead 2-5x: Good - acceptable for most use cases")
    print("  - Overhead > 5x: Consider optimizing hot paths")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
