import autograd.numpy as np
import nvxpy as nvx


def f(A, X, b, z):
    return z @ (A @ np.linalg.norm(X, axis=1) + b)


nvx_func = nvx.Function(f, jac="autograd")


A = nvx.Variable(shape=(10, 7))
X = nvx.Variable(shape=(7, 3))
b = nvx.Variable(shape=(10,))
z = nvx.Variable(shape=(10,))

A.value = np.random.uniform(-1, 1, (10, 7))
X.value = np.random.uniform(-1, 1, (7, 3))
b.value = np.random.uniform(-1, 1, (10,))
z.value = np.random.uniform(-1, 1, (10,))


obj = nvx.sum(nvx_func(A, X, b, z))
cons = [
    nvx.norm(X, ord="fro") >= 10.0,
    z - b <= 0.0,
    obj >= 0.0,
]
prob = nvx.Problem(nvx.Minimize(obj), cons)

prob.solve(solver=nvx.SLSQP)

print(f"norm(X): {np.linalg.norm(X.value, ord='fro')}")
print(f"z - b: {z.value - b.value}")

print(f"final objective: {obj.value}")
print(f"successfully solved? {prob.status}")
