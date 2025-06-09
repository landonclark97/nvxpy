import pytest
from nvxpy.variable import Variable
from nvxpy.problem import Problem, Minimize
from nvxpy.constraint import Constraint
from nvxpy.set import Set
from nvxpy.expression import Expr

def test_variable_creation():
    var = Variable(shape=(2, 2), name="A", symmetric=True)
    assert var.name == "A"
    assert var.shape == (2, 2)
    assert len(var.constraints) > 0

    _ = Variable(shape=(2, 2), name="A", PSD=True)
    _ = Variable(shape=(2, 2), name="A", NSD=True)
    _ = Variable(shape=(2, 2), name="A", pos=True)
    _ = Variable(shape=(2, 2), name="A", neg=True)


def test_problem_creation():
    var = Variable(shape=(2, 2), name="A")
    expr = Expr("add", var, var)
    problem = Problem(Minimize(expr), [var >= 0])
    assert problem.objective.expr.op == "add"
    assert len(problem.constraints) > 0


def test_constraint():
    var = Variable(shape=(1,), name="x")
    constraint = Constraint(var, ">=", 0)
    assert constraint.op == ">="
    assert constraint.left == var


def test_set():
    class CustomSet(Set):
        def constrain(self, var):
            return Constraint(var, ">=", 0)

    custom_set = CustomSet("custom")
    var = Variable(shape=(1,), name="x")
    constraint = custom_set.constrain(var)
    assert constraint.op == ">="

    with pytest.raises(NotImplementedError):
        Set("important set").constrain(None)


def test_expression():
    var1 = Variable(shape=(1,), name="x")
    var2 = Variable(shape=(1,), name="y")
    expr = Expr("add", var1, var2)
    assert expr.op == "add"
    assert expr.left == var1
    assert expr.right == var2
