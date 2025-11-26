"""
Performance comparison: nvxpy interpreted vs compiled vs raw NumPy.

This example benchmarks expression evaluation across three modes:
1. Interpreted (tree-walking evaluation)
2. Compiled (code-generated Python function)
3. Raw NumPy (hand-written equivalent)
"""

import autograd.numpy as np
import nvxpy as nvx
from nvxpy.compiler import compile_to_function
from nvxpy.parser import eval_expression
import time

loops = 100

# Create variables
n = 20  # Size of matrices
A = nvx.Variable(shape=(n, n), name="A")
B = nvx.Variable(shape=(n, n), name="B")
C = nvx.Variable(shape=(n, n), name="C")

# Initialize with random values
A.value = np.random.uniform(-1, 1, (n, n))
B.value = np.random.uniform(-1, 1, (n, n))
C.value = np.random.uniform(-1, 1, (n, n))


def func(A, B, C):
    return (
        np.linalg.norm(A @ B - C, ord="fro") ** 2
        + np.trace(A @ B.T @ C) ** 2
        + np.linalg.norm(A * B + C, ord="fro") ** 2
    )


nvx_func = nvx.Function(func)


# Create a complex expression using nvxpy
complex_expr = 0.0
for _ in range(loops):
    complex_expr += sum(
        nvx.norm(A @ B - C, ord="fro") ** 2
        + nvx.trace(A @ B.T @ C) ** 2
        + nvx.norm(A * B + C, ord="fro") ** 2
        + nvx_func(A, B, C)
        for _ in range(loops)
    )


# Compile the expression for faster evaluation
print("Compiling expression...")
compiled_func = compile_to_function(complex_expr)


# Equivalent NumPy expression
def numpy_equivalent(A, B, C):
    out = 0.0
    for _ in range(loops):
        out += sum(
            np.linalg.norm(A @ B - C, ord="fro") ** 2
            + np.trace(A @ B.T @ C) ** 2
            + np.linalg.norm(A * B + C, ord="fro") ** 2
            + func(A, B, C)
            for _ in range(loops)
        )
    return out


# Warm up runs
print("Warming up...")
var_dict = {"A": A.value, "B": B.value, "C": C.value}
for _ in range(3):
    _ = eval_expression(complex_expr, var_dict)  # Interpreted
    _ = compiled_func(var_dict)  # Compiled
    _ = numpy_equivalent(A.value, B.value, C.value)  # NumPy

# Performance comparison
num_runs = 10
interp_times = []
compiled_times = []
numpy_times = []

print("\nRunning performance comparison...")
print("-" * 70)

for i in range(num_runs):
    var_dict = {"A": A.value, "B": B.value, "C": C.value}

    # Time interpreted evaluation
    start_time = time.time()
    interp_result = eval_expression(complex_expr, var_dict)
    interp_time = time.time() - start_time
    interp_times.append(interp_time)

    # Time compiled evaluation
    start_time = time.time()
    compiled_result = compiled_func(var_dict)
    compiled_time = time.time() - start_time
    compiled_times.append(compiled_time)

    # Time NumPy evaluation
    start_time = time.time()
    numpy_result = numpy_equivalent(A.value, B.value, C.value)
    numpy_time = time.time() - start_time
    numpy_times.append(numpy_time)

    print(
        f"Run {i+1:2d}/{num_runs} - "
        f"Interpreted: {interp_time:.4f}s, "
        f"Compiled: {compiled_time:.4f}s, "
        f"NumPy: {numpy_time:.4f}s"
    )

# Calculate statistics
avg_interp = sum(interp_times) / num_runs
avg_compiled = sum(compiled_times) / num_runs
avg_numpy = sum(numpy_times) / num_runs

print("\n" + "=" * 70)
print("PERFORMANCE RESULTS")
print("=" * 70)
print(f"Matrix size: {n}x{n}")
print(f"Expression complexity: {loops * loops} nested operations")
print(f"Number of runs: {num_runs}")
print()
print(f"Average interpreted time: {avg_interp:.6f} seconds")
print(f"Average compiled time:    {avg_compiled:.6f} seconds")
print(f"Average NumPy time:       {avg_numpy:.6f} seconds")
print()
print(f"Compiled vs Interpreted speedup: {avg_interp / avg_compiled:.2f}x")
print(f"Compiled vs NumPy overhead:      {avg_compiled / avg_numpy:.2f}x")
print(f"Compiled efficiency:             {avg_numpy / avg_compiled * 100:.1f}%")
print()
print(f"Results match (interp vs numpy):    {np.allclose(interp_result, numpy_result)}")
print(f"Results match (compiled vs numpy):  {np.allclose(compiled_result, numpy_result)}")
