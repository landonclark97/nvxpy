import numpy as np

import nvxpy as nvx

BATCH_SIZE = 1000

layers = [1, 10, 10, 10, 10, 1]

W = []
b = []

for layer_in, layer_out in zip(layers[:-1], layers[1:]):
    W.append(nvx.Variable((layer_out, layer_in)))
    b.append(nvx.Variable((layer_out,)))

    W[-1].value = np.random.uniform(-1, 1, (layer_out, layer_in))
    b[-1].value = np.random.uniform(-1, 1, (layer_out,))

def forward(X, W=W, b=b):
    out = X
    for W_i, b_i in zip(W[:-1], b[:-1]):
        out = np.maximum(out @ W_i.value.T + b_i.value, 0)
    out = out @ W[-1].value.T + b[-1].value
    return out

X = np.linspace(-10, 10, BATCH_SIZE).reshape(-1, 1)
Y = np.sin(X) + 0.1 * (X**2)

obj = 0.0
out = X
for W_i, b_i in zip(W[:-1], b[:-1]):
    out = nvx.maximum(out @ W_i.T + b_i, 0)
out = out @ W[-1].T + b[-1]
    
lam = 1e-4
obj += nvx.norm(out - Y, ord='fro') ** 2 + lam * sum([nvx.norm(W_i, ord='fro') for W_i in W])

prob = nvx.Problem(nvx.Minimize(obj))

prob.solve(solver=nvx.LBFGSB)

print(prob.status)
print(prob.solver_stats)


import matplotlib.pyplot as plt

plt.plot(X, forward(X), label='forward')
plt.plot(X, Y, label='true')
# plt.plot(np.abs(forward(X) - Y), label='Error')
plt.legend()
plt.show()