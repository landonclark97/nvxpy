import numpy as np

import nvxpy as nvx


x = nvx.Variable((2,))
x.value = np.array([3.0, 1e-2])

obj = nvx.norm(x - np.array([-3.0, 0.0]))

cons = [nvx.norm(x) >= 1.0]

prob = nvx.Problem(nvx.Minimize(obj), cons)

prob.solve(solver=nvx.SLSQP)

print(prob.status)
print(prob.solver_stats)

print(x.value)
