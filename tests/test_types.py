import pytest
import autograd.numpy as np
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

    _ = Variable(shape=(2, 2), name="B", PSD=True)
    _ = Variable(shape=(2, 2), name="C", NSD=True)
    _ = Variable(shape=(2, 2), name="D", pos=True)
    _ = Variable(shape=(2, 2), name="E", neg=True)


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


def test_variable_functions():
    X = Variable(shape=(2,2), name="x")
    Y = Variable(shape=(2,2), name="y")

    Z = np.zeros((2,2))

    dummy_set = Set("dummy")
    dummy_set.constrain = lambda x: Constraint(x, ">=", 0)

    assert isinstance(X.T, Expr)
    assert isinstance(Z + X, Expr)
    assert isinstance(X + Y, Expr)
    assert isinstance(X - Y, Expr)
    assert isinstance(Z - X, Expr)
    assert isinstance(X * Y, Expr)
    assert isinstance(Z * X, Expr)
    assert isinstance(X / Y, Expr)
    assert isinstance(Z @ X, Expr)
    assert isinstance(X @ Z, Expr)
    assert isinstance(X**2, Expr)
    assert isinstance(X**Y, Expr)
    assert isinstance(-X, Expr)
    assert isinstance(X[0], Expr)
    assert isinstance(X >= Y, Constraint)
    assert isinstance(X <= Y, Constraint)
    assert isinstance(X == Y, Constraint)
    assert isinstance(X >> Y, Constraint)
    assert isinstance(X << Y, Constraint)
    assert isinstance(X ^ dummy_set, Constraint)

    assert X.value is None
    X.value = Z
    assert np.allclose(X.value, Z)

    Y.value = Z
    assert np.allclose(X.value + Y.value, Z)


def test_expression_functions():
    A = Variable(shape=(2,2), name="x")
    Y = Variable(shape=(2,2), name="y")

    X = A + Y

    Z = np.zeros((2,2))

    dummy_set = Set("dummy")
    dummy_set.constrain = lambda x: Constraint(x, ">=", 0)

    assert isinstance(X.T, Expr)
    assert isinstance(Z + X, Expr)
    assert isinstance(X + Y, Expr)
    assert isinstance(X - Y, Expr)
    assert isinstance(Z - X, Expr)
    assert isinstance(X * Y, Expr)
    assert isinstance(Z * X, Expr)
    assert isinstance(X / Y, Expr)
    assert isinstance(Z @ X, Expr)
    assert isinstance(X @ Z, Expr)
    assert isinstance(X**2, Expr)
    assert isinstance(X**Y, Expr)
    assert isinstance(-X, Expr)
    assert isinstance(X[0], Expr)
    assert isinstance(X >= Y, Constraint)
    assert isinstance(X <= Y, Constraint)
    assert isinstance(X == Y, Constraint)
    assert isinstance(X >> Y, Constraint)
    assert isinstance(X << Y, Constraint)
    assert isinstance(X ^ dummy_set, Constraint)


def test_shape_handling():
    # Test basic variable shapes
    x = Variable(shape=(3, 4), name="x")
    assert x.shape == (3, 4)
    
    y = Variable(shape=(4, 2), name="y")
    assert y.shape == (4, 2)
    
    # Test matrix multiplication shapes
    z = x @ y
    assert z.shape == (3, 2)
    
    # Test matrix multiplication with constants
    const_mat = np.ones((4, 2))
    assert (x @ const_mat).shape == (3, 2)
    assert (const_mat.T @ x.T).shape == (2, 3)
    
    # Test elementwise operations shape with broadcasting
    a = Variable(shape=(2, 2), name="a")
    b = Variable(shape=(2, 2), name="b")
    c = Variable(shape=(1, 2), name="c")
    d = Variable(shape=(2, 1), name="d")
    e = Variable(shape=(1, 1), name="e")
    
    # Same shape operations
    assert (a + b).shape == (2, 2)
    assert (a - b).shape == (2, 2)
    assert (a * b).shape == (2, 2)
    assert (a / b).shape == (2, 2)
    
    # Broadcasting operations - expanding dimensions
    assert (c + a).shape == (2, 2)  # (1,2) broadcasts to (2,2)
    assert (d * b).shape == (2, 2)  # (2,1) broadcasts to (2,2)
    assert (c * d).shape == (2, 2)  # (1,2) and (2,1) broadcast to (2,2)
    assert (e + a).shape == (2, 2)  # (1,1) broadcasts to (2,2)
    
    # Broadcasting with constants
    const_scalar = 2.0
    const_vec_h = np.ones((1, 2))
    const_vec_v = np.ones((2, 1))
    const_mat = np.ones((2, 2))
    
    assert (const_scalar * a).shape == (2, 2)
    assert (a * const_scalar).shape == (2, 2)
    assert (const_vec_h * a).shape == (2, 2)
    assert (a * const_vec_v).shape == (2, 2)
    assert (const_mat + a).shape == (2, 2)
    assert (a - const_mat).shape == (2, 2)
    
    # Test division with constants
    assert (a / const_scalar).shape == (2, 2)
    assert (a / const_vec_h).shape == (2, 2)
    assert (a / const_vec_v).shape == (2, 2)
    assert (const_mat / a).shape == (2, 2)
    
    # Test power operations
    assert (a ** 2).shape == (2, 2)
    assert (a ** const_scalar).shape == (2, 2)
    assert (e ** 2).shape == (1, 1)  # Preserves (1,1) shape
    
    # Test transpose shapes
    assert a.T.shape == (2, 2)
    assert c.T.shape == (2, 1)  # (1,2) -> (2,1)
    assert d.T.shape == (1, 2)  # (2,1) -> (1,2)
    assert e.T.shape == (1, 1)  # (1,1) -> (1,1)
    
    # Test shape compatibility errors
    with pytest.raises(ValueError):
        expr = a @ z  # Incompatible dimensions for matmul (2,2) @ (3,2)
        expr.shape
    
    with pytest.raises(ValueError):
        expr = a + Variable(shape=(3, 3), name="wrong_size")  # Non-broadcastable shapes
        expr.shape
        
    with pytest.raises(ValueError):
        expr = a @ Variable(shape=(3, 2), name="wrong_matmul")  # Wrong inner dimensions
        expr.shape
        
    with pytest.raises(ValueError):
        expr = Variable(shape=(2, 3), name="m1") @ Variable(shape=(2, 2), name="m2")  # Wrong inner dimensions
        expr.shape
        
    with pytest.raises(ValueError):
        expr = a + Variable(shape=(3, 1), name="wrong_broadcast")  # Non-broadcastable shapes
        expr.shape
        
    # Test right-hand operations with constants
    assert (const_scalar + a).shape == (2, 2)  # right add
    assert (const_mat - a).shape == (2, 2)  # right subtract
    assert (const_vec_h * a).shape == (2, 2)  # right multiply
    assert (const_vec_v / a).shape == (2, 2)  # right divide