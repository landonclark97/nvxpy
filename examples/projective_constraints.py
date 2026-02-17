import nvxpy as nvx
import numpy as np

X = nvx.Variable(shape=(5, 5))
X.value = np.random.uniform(-1, 1, (5, 5))
Y = nvx.Variable(shape=(5, 5))
Y.value = np.random.uniform(-1, 1, (5, 5))

R1 = np.eye(5) + np.random.uniform(-1, 1, (5, 5))
R2 = np.eye(5) + np.random.uniform(-1, 1, (5, 5))

obj = nvx.norm(X - R1, ord="fro") + nvx.norm(Y - R2, ord="fro")
print(f"initial objective: {obj.value}")

cons = [X + Y ^ nvx.SO(5)]
problem = nvx.Problem(nvx.Minimize(obj), cons)
problem.solve(solver_options={"p_tol": 1e-2, "maxiter": 100})

print(f"final objective: {obj.value}")

Z = X + Y
print(f"orthogonality error: {np.amax(Z.value.T @ Z.value - np.eye(5))}")
print(f"determinant error: {np.linalg.det(Z.value)}")

print(f"SUCCESS: {problem.status}")
print(problem.solver_stats)
