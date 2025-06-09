import pytest
import autograd.numpy as np
import nvxpy as nvx
from nvxpy.sets.special_orthogonal import SO
from nvxpy.constraint import Constraint
from nvxpy.atoms.polar import PolarDecomposition
from nvxpy.problem import Problem, Minimize
from nvxpy.variable import Variable


def test_so_initialization():
    n = 3
    so = SO(n)
    assert so.n == n
    assert so.name == f"SO({n})"


def test_so_constrain():
    n = 3
    so = SO(n)
    var = Variable(shape=(n, n), name="some_variable")
    constraint = so.constrain(var)
    assert isinstance(constraint, Constraint)
    assert constraint.op == "=="
    assert constraint.right == PolarDecomposition(var)


def test_so_constraint_w_problem():
    n = 3
    so_n = SO(n)
    var = Variable(shape=(n, n), name="some_variable")
    var.value = np.random.uniform(-2, 2, (n, n))
    obj = nvx.norm(var - np.eye(n), ord='fro')
    problem = Problem(Minimize(obj), [var ^ so_n])
    problem.solve()

    assert np.allclose(var.value.T @ var.value, np.eye(n))