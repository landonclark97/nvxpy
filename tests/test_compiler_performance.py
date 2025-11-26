"""
Performance Benchmark Suite for Expression Evaluation

This module compares the performance of:
1. Original recursive eval_expression (interpreter)
2. Compiled expression evaluation (compiler)

Test cases range from small to large expressions to measure scaling behavior.
"""

import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# Import nvxpy components
import sys
sys.path.insert(0, "src")

import nvxpy as nvx
from nvxpy.parser import eval_expression
from nvxpy.compiler import compile_expression, eval_compiled, compile_to_function
from nvxpy.variable import Variable


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    num_nodes: int
    interpreted_time: float
    compiled_time: float
    compile_overhead: float
    codegen_time: float
    codegen_overhead: float
    speedup_ir: float
    speedup_codegen: float
    results_match: bool

    def __str__(self):
        return (
            f"{self.name:40s} | "
            f"nodes: {self.num_nodes:5d} | "
            f"interp: {self.interpreted_time*1000:7.3f}ms | "
            f"IR: {self.compiled_time*1000:7.3f}ms ({self.speedup_ir:5.2f}x) | "
            f"codegen: {self.codegen_time*1000:7.3f}ms ({self.speedup_codegen:5.2f}x) | "
            f"match: {self.results_match}"
        )


def reset_variable_ids():
    """Reset variable ID counter between tests."""
    Variable._ids = 0
    Variable._used_names = set()


def count_nodes(expr) -> int:
    """Count the number of nodes in an expression tree."""
    if isinstance(expr, Variable):
        return 1
    elif isinstance(expr, nvx.expression.Expr):
        count = 1
        count += count_nodes(expr.left)
        if expr.right is not None:
            count += count_nodes(expr.right)
        return count
    elif hasattr(expr, 'args') and expr.args:  # Function
        return 1 + sum(count_nodes(arg) for arg in expr.args)
    else:
        return 0  # Constants don't count


def benchmark_expression(
    name: str,
    expr,
    var_dict: Dict[str, np.ndarray],
    num_evals: int = 100,
) -> BenchmarkResult:
    """
    Benchmark an expression with all evaluation methods.

    Args:
        name: Name of the benchmark
        expr: Expression to evaluate
        var_dict: Variable values
        num_evals: Number of evaluations for timing

    Returns:
        BenchmarkResult with timing and correctness info
    """
    num_nodes = count_nodes(expr)

    # Warm up and get reference result
    ref_result = eval_expression(expr, var_dict)

    # Time interpreted evaluation
    start = time.perf_counter()
    for _ in range(num_evals):
        eval_expression(expr, var_dict)
    interpreted_time = (time.perf_counter() - start) / num_evals

    # Time IR compilation
    compile_start = time.perf_counter()
    compiled = compile_expression(expr)
    compile_overhead = time.perf_counter() - compile_start

    # Time IR compiled evaluation
    start = time.perf_counter()
    for _ in range(num_evals):
        compiled_result = eval_compiled(compiled, var_dict)
    compiled_time = (time.perf_counter() - start) / num_evals

    # Time codegen compilation
    codegen_start = time.perf_counter()
    codegen_func = compile_to_function(expr)
    codegen_overhead = time.perf_counter() - codegen_start

    # Time codegen evaluation
    start = time.perf_counter()
    for _ in range(num_evals):
        codegen_result = codegen_func(var_dict)
    codegen_time = (time.perf_counter() - start) / num_evals

    # Check results match
    results_match = (
        np.allclose(ref_result, compiled_result, rtol=1e-10, atol=1e-10) and
        np.allclose(ref_result, codegen_result, rtol=1e-10, atol=1e-10)
    )

    speedup_ir = interpreted_time / compiled_time if compiled_time > 0 else float('inf')
    speedup_codegen = interpreted_time / codegen_time if codegen_time > 0 else float('inf')

    return BenchmarkResult(
        name=name,
        num_nodes=num_nodes,
        interpreted_time=interpreted_time,
        compiled_time=compiled_time,
        compile_overhead=compile_overhead,
        codegen_time=codegen_time,
        codegen_overhead=codegen_overhead,
        speedup_ir=speedup_ir,
        speedup_codegen=speedup_codegen,
        results_match=results_match,
    )


# =============================================================================
# BENCHMARK SUITE
# =============================================================================

def bench_tiny_scalar():
    """Tiny: Single scalar operation."""
    reset_variable_ids()
    x = nvx.Variable((1,), name="x")
    expr = x + 1

    var_dict = {"x": np.array([5.0])}
    return benchmark_expression("tiny_scalar (x + 1)", expr, var_dict, num_evals=1000)


def bench_small_vector():
    """Small: Simple vector operations."""
    reset_variable_ids()
    x = nvx.Variable((10,), name="x")
    y = nvx.Variable((10,), name="y")
    expr = x + y * 2 - 3

    var_dict = {
        "x": np.random.randn(10),
        "y": np.random.randn(10),
    }
    return benchmark_expression("small_vector (x + y*2 - 3)", expr, var_dict, num_evals=1000)


def bench_medium_matrix():
    """Medium: Matrix operations."""
    reset_variable_ids()
    A = nvx.Variable((10, 10), name="A")
    B = nvx.Variable((10, 10), name="B")
    expr = A @ B + A.T - B * 2

    var_dict = {
        "A": np.random.randn(10, 10),
        "B": np.random.randn(10, 10),
    }
    return benchmark_expression("medium_matrix (A@B + A.T - B*2)", expr, var_dict, num_evals=500)


def bench_medium_with_norm():
    """Medium: Expression with norm atom."""
    reset_variable_ids()
    x = nvx.Variable((20,), name="x")
    y = nvx.Variable((20,), name="y")
    expr = nvx.norm(x - y) ** 2 + nvx.norm(x + y)

    var_dict = {
        "x": np.random.randn(20),
        "y": np.random.randn(20),
    }
    return benchmark_expression("medium_with_norm", expr, var_dict, num_evals=500)


def bench_medium_with_trace():
    """Medium: Expression with trace atom."""
    reset_variable_ids()
    A = nvx.Variable((10, 10), name="A")
    B = nvx.Variable((10, 10), name="B")
    expr = nvx.trace(A @ B.T) + nvx.trace(A.T @ A)

    var_dict = {
        "A": np.random.randn(10, 10),
        "B": np.random.randn(10, 10),
    }
    return benchmark_expression("medium_with_trace", expr, var_dict, num_evals=500)


def bench_large_chain():
    """Large: Long chain of operations."""
    reset_variable_ids()
    x = nvx.Variable((50,), name="x")

    # Build a chain: x + x*2 + x*3 + ... + x*N
    expr = x
    for i in range(2, 51):
        expr = expr + x * i

    var_dict = {"x": np.random.randn(50)}
    return benchmark_expression("large_chain (50 ops)", expr, var_dict, num_evals=200)


def bench_large_tree():
    """Large: Deep binary tree of operations."""
    reset_variable_ids()

    # Create 8 variables
    vars = [nvx.Variable((20,), name=f"v{i}") for i in range(8)]
    var_dict = {f"v{i}": np.random.randn(20) for i in range(8)}

    # Build tree: ((v0+v1)*(v2+v3)) + ((v4-v5)*(v6-v7))
    left = (vars[0] + vars[1]) * (vars[2] + vars[3])
    right = (vars[4] - vars[5]) * (vars[6] - vars[7])
    expr = left + right

    return benchmark_expression("large_tree (8 vars, balanced)", expr, var_dict, num_evals=500)


def bench_large_matrix_expr():
    """Large: Complex matrix expression."""
    reset_variable_ids()
    A = nvx.Variable((20, 20), name="A")
    B = nvx.Variable((20, 20), name="B")
    C = nvx.Variable((20, 20), name="C")

    # (A @ B + B @ C) * (A.T @ C - B.T @ A)
    expr = (A @ B + B @ C) * (A.T @ C - B.T @ A)

    var_dict = {
        "A": np.random.randn(20, 20),
        "B": np.random.randn(20, 20),
        "C": np.random.randn(20, 20),
    }
    return benchmark_expression("large_matrix_expr (3 vars)", expr, var_dict, num_evals=200)


def bench_xlarge_polynomial():
    """XLarge: High-degree polynomial."""
    reset_variable_ids()
    x = nvx.Variable((100,), name="x")

    # Build polynomial: x + x^2 + x^3 + ... + x^20
    expr = x
    for i in range(2, 21):
        expr = expr + x ** i

    var_dict = {"x": np.random.randn(100) * 0.5}  # Small values to avoid overflow
    return benchmark_expression("xlarge_polynomial (deg 20)", expr, var_dict, num_evals=100)


def bench_xlarge_multi_norm():
    """XLarge: Multiple norm operations."""
    reset_variable_ids()

    vars = [nvx.Variable((30,), name=f"x{i}") for i in range(10)]
    var_dict = {f"x{i}": np.random.randn(30) for i in range(10)}

    # Sum of pairwise norm differences
    expr = nvx.norm(vars[0] - vars[1]) ** 2
    for i in range(1, 9):
        expr = expr + nvx.norm(vars[i] - vars[i+1]) ** 2

    return benchmark_expression("xlarge_multi_norm (10 vars)", expr, var_dict, num_evals=200)


def bench_xlarge_matrix_chain():
    """XLarge: Chain of matrix multiplications."""
    reset_variable_ids()

    vars = [nvx.Variable((15, 15), name=f"M{i}") for i in range(6)]
    var_dict = {f"M{i}": np.random.randn(15, 15) * 0.3 for i in range(6)}

    # M0 @ M1 @ M2 @ M3 @ M4 @ M5
    expr = vars[0]
    for i in range(1, 6):
        expr = expr @ vars[i]

    return benchmark_expression("xlarge_matrix_chain (6 mats)", expr, var_dict, num_evals=100)


def bench_xxlarge_sum_of_norms():
    """XXLarge: Many norm operations."""
    reset_variable_ids()

    n_vars = 20
    vars = [nvx.Variable((50,), name=f"x{i}") for i in range(n_vars)]
    var_dict = {f"x{i}": np.random.randn(50) for i in range(n_vars)}

    # Sum of all squared norms
    expr = nvx.norm(vars[0]) ** 2
    for i in range(1, n_vars):
        expr = expr + nvx.norm(vars[i]) ** 2

    return benchmark_expression("xxlarge_sum_of_norms (20 vars)", expr, var_dict, num_evals=100)


def bench_xxlarge_combined():
    """XXLarge: Combined matrix and vector operations."""
    reset_variable_ids()

    A = nvx.Variable((30, 30), name="A")
    B = nvx.Variable((30, 30), name="B")
    x = nvx.Variable((30,), name="x")
    y = nvx.Variable((30,), name="y")

    # Complex expression with matrices and vectors
    mat_expr = A @ B + B @ A.T - A ** 2
    vec_expr = (A @ x - B @ y) + (B.T @ x + A.T @ y)
    expr = nvx.trace(mat_expr) + nvx.norm(vec_expr) ** 2

    var_dict = {
        "A": np.random.randn(30, 30),
        "B": np.random.randn(30, 30),
        "x": np.random.randn(30),
        "y": np.random.randn(30),
    }
    return benchmark_expression("xxlarge_combined (mats+vecs)", expr, var_dict, num_evals=100)


def bench_indexing():
    """Test indexing operations."""
    reset_variable_ids()

    A = nvx.Variable((20, 20), name="A")
    var_dict = {"A": np.random.randn(20, 20)}

    # Simple indexing operations (compatible shapes)
    expr = A[0, :] + A[:, 0] + A[1, :]

    return benchmark_expression("indexing_ops", expr, var_dict, num_evals=500)


def bench_nested_atoms():
    """Test nested atom operations."""
    reset_variable_ids()

    A = nvx.Variable((15, 15), name="A")
    B = nvx.Variable((15, 15), name="B")
    var_dict = {
        "A": np.random.randn(15, 15),
        "B": np.random.randn(15, 15),
    }

    # Nested atoms: norm of traced matrices
    expr = nvx.norm(A @ B, ord="fro") ** 2 + nvx.trace(A @ B.T) ** 2

    return benchmark_expression("nested_atoms", expr, var_dict, num_evals=200)


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    benchmarks = [
        bench_tiny_scalar,
        bench_small_vector,
        bench_medium_matrix,
        bench_medium_with_norm,
        bench_medium_with_trace,
        bench_large_chain,
        bench_large_tree,
        bench_large_matrix_expr,
        bench_xlarge_polynomial,
        bench_xlarge_multi_norm,
        bench_xlarge_matrix_chain,
        bench_xxlarge_sum_of_norms,
        bench_xxlarge_combined,
        bench_indexing,
        bench_nested_atoms,
    ]

    print("=" * 120)
    print("NVXPY Expression Evaluation Benchmark Suite")
    print("=" * 120)
    print()

    results = []
    for bench_fn in benchmarks:
        try:
            result = bench_fn()
            results.append(result)
            print(result)
        except Exception as e:
            print(f"FAILED: {bench_fn.__name__}: {e}")

    print()
    print("=" * 140)
    print("SUMMARY")
    print("=" * 140)

    # Calculate aggregates
    total_interp = sum(r.interpreted_time for r in results)
    total_ir = sum(r.compiled_time for r in results)
    total_codegen = sum(r.codegen_time for r in results)
    avg_speedup_ir = sum(r.speedup_ir for r in results) / len(results)
    avg_speedup_codegen = sum(r.speedup_codegen for r in results) / len(results)
    all_match = all(r.results_match for r in results)

    print(f"Total interpreted time:   {total_interp*1000:.3f}ms")
    print(f"Total IR compiled time:   {total_ir*1000:.3f}ms  (avg {avg_speedup_ir:.2f}x speedup)")
    print(f"Total codegen time:       {total_codegen*1000:.3f}ms  (avg {avg_speedup_codegen:.2f}x speedup)")
    print(f"All results match:        {all_match}")
    print()

    # Speedup breakdown by size category
    print("Speedup by expression size (codegen):")
    small = [r for r in results if r.num_nodes <= 10]
    medium = [r for r in results if 10 < r.num_nodes <= 50]
    large = [r for r in results if r.num_nodes > 50]

    if small:
        print(f"  Small (<=10 nodes):   {sum(r.speedup_codegen for r in small)/len(small):.2f}x avg")
    if medium:
        print(f"  Medium (11-50 nodes): {sum(r.speedup_codegen for r in medium)/len(medium):.2f}x avg")
    if large:
        print(f"  Large (>50 nodes):    {sum(r.speedup_codegen for r in large)/len(large):.2f}x avg")

    return results


def run_scaling_analysis():
    """Analyze how speedup scales with expression size."""
    print()
    print("=" * 140)
    print("SCALING ANALYSIS")
    print("=" * 140)
    print()

    reset_variable_ids()

    # Generate expressions of increasing size
    sizes = [5, 10, 20, 50, 100, 200, 500]
    results = []

    for n in sizes:
        reset_variable_ids()
        x = nvx.Variable((20,), name="x")

        # Build chain of n operations
        expr = x
        for i in range(2, n + 1):
            expr = expr + x * (i % 10 + 1)  # Vary multiplier to avoid trivial patterns

        var_dict = {"x": np.random.randn(20)}
        result = benchmark_expression(f"chain_size_{n}", expr, var_dict, num_evals=100)
        results.append(result)
        print(result)

    print()
    print("Scaling trend (codegen speedup):")
    for r in results:
        bar = "█" * int(r.speedup_codegen * 3)
        print(f"  {r.num_nodes:4d} nodes: {bar} {r.speedup_codegen:.2f}x")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run expression evaluation benchmarks")
    parser.add_argument("--scaling", action="store_true", help="Run scaling analysis")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.scaling:
        run_scaling_analysis()
    elif args.all:
        run_all_benchmarks()
        run_scaling_analysis()
    else:
        run_all_benchmarks()
