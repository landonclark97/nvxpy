import numpy as np

import nvxpy as nvx
from nvxpy.sets import SO




X = nvx.Variable(shape=(3, 3))
X.value = np.random.uniform(-1, 1, (3, 3))

obj = nvx.det(X) + nvx.norm(X, ord='fro') + nvx.sum(X)

cons = [X ^ SO(3)]

prob = nvx.Problem(nvx.Minimize(obj), cons)

prob.solve(solver=nvx.SLSQP)

print(prob.status)
print(prob.solver_stats)

print(X.value.T @ X.value)



x = nvx.Variable((2,))
x.value = np.array([3.0, 1e-2])

obj = nvx.norm(x - np.array([-3.0, 0.0]))

cons = [nvx.norm(x) >= 1.0]

prob = nvx.Problem(nvx.Minimize(obj), cons)

prob.solve(solver=nvx.SLSQP)

print(prob.status)
print(prob.solver_stats)

print(x.value)


R = nvx.Variable((3, 3))
R.value = np.random.uniform(-1, 1, (3, 3))

obj = nvx.norm(R - np.eye(3), ord='fro')

cons = [R ^ SO(3)]

prob = nvx.Problem(nvx.Maximize(obj), cons)

prob.solve(solver=nvx.TRUST_CONSTR)

print(prob.status)
print(prob.solver_stats)

print(R.value)
