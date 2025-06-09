import autograd.numpy as np

import nvxpy as nvx


BATCH_SIZE = 1000

W = []  # weights
b = []  # biases

layers = [1, 10, 10, 10, 10, 1]  # number of neurons in each layer
for layer_in, layer_out in zip(layers[:-1], layers[1:]):
    W.append(nvx.Variable((layer_out, layer_in)))
    b.append(nvx.Variable((layer_out,)))

    W[-1].value = np.random.uniform(-1, 1, (layer_out, layer_in))
    b[-1].value = np.random.uniform(-1, 1, (layer_out,))


# forward pass of neural network
def forward(X, W, b, activation=lambda x: np.maximum(x, 0)):
    out = X
    for W_i, b_i in zip(W[:-1], b[:-1]):
        out = activation(out @ W_i.T + b_i)
    out = out @ W[-1].T + b[-1]
    return out

nvx_forward = nvx.Function(forward, jac="autograd")

# generate data
X = np.linspace(-10, 10, BATCH_SIZE).reshape(-1, 1)
Y = np.sin(X) + 0.1 * (X**2)

# loss function
Y_hat = nvx_forward(X, W, b, activation=np.tanh)
training_error = nvx.norm(Y_hat - Y, ord="fro") ** 2
regularization = sum([nvx.norm(W_i, ord="fro") for W_i in W])

# combine terms, weighted multi-objective optimization
lam = 1e-4
obj = training_error + lam * regularization

# train model
prob = nvx.Problem(nvx.Minimize(obj))
prob.solve(solver=nvx.LBFGSB)

print(prob.status)
print(prob.solver_stats)

print(f"training error: {training_error.value}")
print(f"L2 term: {regularization.value}")
