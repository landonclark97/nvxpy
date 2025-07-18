import autograd.numpy as np
import nvxpy as nvx
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
    return np.linalg.norm(A @ B - C, ord="fro") ** 2 + np.trace(A @ B.T @ C) ** 2 + np.linalg.norm(A * B + C, ord="fro") ** 2
nvx_func = nvx.Function(func)


# Create a complex expression using nvxpy
complex_expr = 0.0
for _ in range(loops):
    complex_expr += sum(
        nvx.norm(A @ B - C, ord="fro") ** 2 +
        nvx.trace(A @ B.T @ C) ** 2 +
        nvx.norm(A * B + C, ord="fro") ** 2 +
        nvx_func(A, B, C)
    for _ in range(loops))

# Equivalent NumPy expression
def numpy_equivalent(A, B, C):
    out = 0.0
    for _ in range(loops):
        out += sum(
            np.linalg.norm(A @ B - C, ord='fro') ** 2 +
            np.trace(A @ B.T @ C) ** 2 +
            np.linalg.norm(A * B + C, ord='fro') ** 2 +
            func(A, B, C)
        for _ in range(loops))
    return out

# Warm up runs
print("Warming up...")
for _ in range(10):
    _ = complex_expr.value
    _ = numpy_equivalent(A.value, B.value, C.value)

# Performance comparison
num_runs = 10
nvx_times = []
numpy_times = []

print("\nRunning performance comparison...")
for i in range(num_runs):
    # Time nvxpy evaluation
    start_time = time.time()
    nvx_result = complex_expr.value
    nvx_time = time.time() - start_time
    nvx_times.append(nvx_time)
    
    # Time NumPy evaluation
    start_time = time.time()
    numpy_result = numpy_equivalent(A.value, B.value, C.value)
    numpy_time = time.time() - start_time
    numpy_times.append(numpy_time)
    
    print(f"Run {i+1}/{num_runs} - nvxpy: {nvx_time:.8f}s, NumPy: {numpy_time:.8f}s")

# Calculate statistics
avg_nvx = sum(nvx_times) / num_runs
avg_numpy = sum(numpy_times) / num_runs
speedup = avg_numpy / avg_nvx

print("\nPerformance Results:")
print(f"Matrix size: {n}x{n}")
print(f"Number of runs: {num_runs}")
print(f"Average nvxpy time: {avg_nvx:.8f} seconds")
print(f"Average NumPy time: {avg_numpy:.8f} seconds")
print(f"Speedup factor: {speedup:.2f}x")
print(f"\nResults match: {np.allclose(nvx_result, numpy_result)}")
