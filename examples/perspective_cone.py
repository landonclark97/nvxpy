import itertools

import autograd.numpy as np

import nvxpy as nvx


Xd = []
for _ in range(4):
    x = nvx.Variable(shape=(4,))
    x.value = np.random.uniform(-1, 1, size=(4,))
    Xd.append(x)

Ps = []
for _ in range(4):
    p = nvx.Variable()
    p.value = np.random.rand()
    Ps.append(p)

Ts = []
for _ in range(4):
    t = nvx.Variable()
    t.value = np.random.rand()
    Ts.append(t)

Xc = []
for idx in range(4):
    center = np.array([2.0, 2.0, 2.0, 2.0])
    center[idx] *= -1
    Xc.append(center)

Xcones = []
for idx in range(4):
    Xcones.append(nvx.PerspectiveCone(nvx.norm, Xd[idx] - Xc[idx], Ps[idx]))

# nonsense optimization problem ¯\_(ツ)_/¯
obj = 0.0
for ti, tj in itertools.combinations(Ts, 2):
    obj += ti - tj

cons = []
for idx in range(4):
    cons.append(Ts[idx] ^ Xcones[idx])
    cons.append(Ps[idx] >= 0)
    cons.append(Ps[idx] <= 1)
    cons.append(nvx.norm(Xd[idx] - Xc[idx]) <= 1)
cons.append(sum(Ps) == 1)

prob = nvx.Problem(nvx.Maximize(obj), cons)
prob.solve(solver=nvx.SLSQP)

print(prob.solver_stats)
print(f"solved successfully? {prob.status}")
print(f"Ts: {[t.value for t in Ts]}")
print(f"Ps: {[p.value for p in Ps]}")
print(f"Xd: {[x.value for x in Xd]}")
print(f"final objective: {obj.value}")
