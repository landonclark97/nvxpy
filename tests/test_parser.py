from nvxpy.parser import collect_vars, eval_expression
from nvxpy.expression import Expr
from nvxpy.variable import Variable


def test_collect_vars():
    # Test with a single variable
    x = Variable(name="x")
    vars = []
    collect_vars(x, vars)
    assert vars == [x]

    # Test with an expression containing variables
    y = Variable(name="y")
    expr = Expr(left=x, right=y, op="add")
    vars = []
    collect_vars(expr, vars)
    assert vars == [x, y]


def test_eval_expression():
    # Test evaluation of a single variable
    x = Variable(name="x")
    x.value = 5
    var_dict = {"x": 10}
    assert eval_expression(x, var_dict) == 10
    assert eval_expression(x, var_dict, use_value=True) == 5

    # Test evaluation of an addition expression
    y = Variable(name="y")
    y.value = 3
    expr = Expr(left=x, right=y, op="add")
    var_dict = {"x": 10, "y": 2}
    assert eval_expression(expr, var_dict) == 12
    assert eval_expression(expr, var_dict, use_value=True) == 8

    # Test evaluation of a multiplication expression
    expr = Expr(left=x, right=y, op="mul")
    assert eval_expression(expr, var_dict) == 20
    assert eval_expression(expr, var_dict, use_value=True) == 15

    # Test evaluation of a negation expression
    expr = Expr(left=x, op="neg")
    assert eval_expression(expr, var_dict) == -10
    assert eval_expression(expr, var_dict, use_value=True) == -5

    # Test evaluation of a power expression
    expr = Expr(left=x, right=y, op="pow")
    assert eval_expression(expr, var_dict) == 100
    assert eval_expression(expr, var_dict, use_value=True) == 125


def test_collect_vars_with_list():
    # Test with a list of variables
    x = Variable(name="x")
    y = Variable(name="y")
    z = Variable(name="z")
    vars = []
    collect_vars([x, y, z], vars)
    assert vars == [x, y, z]


def test_collect_vars_with_dict():
    # Test with a dictionary of variables
    x = Variable(name="x")
    y = Variable(name="y")
    z = Variable(name="z")
    vars = []
    collect_vars({"x": x, "y": y, "z": z}, vars)
    assert vars == [x, y, z]


def test_eval_expression_with_list():
    # Test evaluation with a list of variables
    x = Variable(name="x")
    y = Variable(name="y")
    z = Variable(name="z")
    x.value = 5
    y.value = 3
    z.value = 2
    var_dict = {"x": 10, "y": 2, "z": 1}
    expr_list = [x, y, z]
    assert eval_expression(expr_list, var_dict) == [10, 2, 1]
    assert eval_expression(expr_list, var_dict, use_value=True) == [5, 3, 2]


def test_eval_expression_with_dict():
    # Test evaluation with a dictionary of variables
    x = Variable(name="x")
    y = Variable(name="y")
    z = Variable(name="z")
    x.value = 5
    y.value = 3
    z.value = 2
    var_dict = {"x": 10, "y": 2, "z": 1}
    expr_dict = {"x": x, "y": y, "z": z}
    assert eval_expression(expr_dict, var_dict) == {"x": 10, "y": 2, "z": 1}
    assert eval_expression(expr_dict, var_dict, use_value=True) == {"x": 5, "y": 3, "z": 2}
