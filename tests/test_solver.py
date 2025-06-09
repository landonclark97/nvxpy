import autograd.numpy as np
import nvxpy as nvx
from nvxpy.problem import Problem, Minimize
from nvxpy.variable import Variable


def test_convex_problem():
    # Define variables
    x = Variable(shape=(2,), name="x")

    # Define objective: Minimize (x1 - 1)^2 + (x2 - 2)^2
    objective = Minimize((x[0] - 1) ** 2 + (x[1] - 2) ** 2)

    # Define constraints: x1 + x2 = 3
    constraints = [x[0] + x[1] == 3]

    # Create and solve problem
    problem = Problem(objective, constraints)
    problem.solve()

    # Assert solution
    assert np.allclose(x.value, np.array([1.0, 2.0]), atol=1e-4)


def test_non_convex_obj():
    # Define variables
    x = Variable(shape=(2,), name="x")

    # Define objective: Minimize sin(x1) + cos(x2)
    def cos_sin_obj(x):
        return np.sin(x[0]) + np.cos(x[1])

    obj_func = nvx.Function(cos_sin_obj, jac="autograd")

    objective = Minimize(obj_func(x))

    # Define constraints: x1^2 + x2^2 <= 1
    constraints = [x[0] ** 2 + x[1] ** 2 <= 1]

    # Create and solve problem
    problem = Problem(objective, constraints)
    problem.solve()

    # Assert solution is within the unit circle
    assert x.value[0] ** 2 + x.value[1] ** 2 <= 1


def test_non_convex_constraints():
    # Define variables
    x = Variable(shape=(2,), name="x")
    x.value = np.array([2.0, 0.0])

    x_d = np.array([-2.0, 0.0])

    objective = Minimize(nvx.norm(x - x_d))

    # Define constraints: x1^2 + x2^2 <= 1
    constraints = [nvx.norm(x) >= 1]

    # Create and solve problem
    problem = Problem(objective, constraints)
    problem.solve()

    # Assert solution is within the unit circle
    assert x.value[0] ** 2 + x.value[1] ** 2 >= 1
    assert not np.allclose(x.value, x_d, atol=1e-4)

    x.value = np.array([0.0, 2.0])
    problem.solve()
    assert x.value[0] ** 2 + x.value[1] ** 2 >= 1
    assert np.allclose(x.value, x_d, atol=1e-4)

    x.value = np.array([0.0, -2.0])
    problem.solve()
    assert x.value[0] ** 2 + x.value[1] ** 2 >= 1
    assert np.allclose(x.value, x_d, atol=1e-4)
