"""
Correctness tests for the expression compiler.

These tests verify that compiled evaluation produces the same results
as the interpreted evaluation across all supported operations.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, "src")

import nvxpy as nvx
from nvxpy.parser import eval_expression
from nvxpy.compiler import (
    compile_expression,
    eval_compiled,
    compile_to_function,
)
from nvxpy.variable import reset_variable_ids


def assert_compiled_matches(expr, var_dict, rtol=1e-10, atol=1e-10):
    """Assert that compiled evaluation matches interpreted evaluation."""
    # Get reference result
    ref_result = eval_expression(expr, var_dict)

    # Test IR compilation
    compiled = compile_expression(expr)
    ir_result = eval_compiled(compiled, var_dict)
    np.testing.assert_allclose(ir_result, ref_result, rtol=rtol, atol=atol,
                               err_msg="IR compiled result doesn't match")

    # Test codegen compilation
    codegen_func = compile_to_function(expr)
    codegen_result = codegen_func(var_dict)
    np.testing.assert_allclose(codegen_result, ref_result, rtol=rtol, atol=atol,
                               err_msg="Codegen compiled result doesn't match")


class TestBasicOperations:
    """Test basic arithmetic operations."""

    def test_add_scalars(self):
        reset_variable_ids()
        x = nvx.Variable((1,), name="x")
        expr = x + 5
        var_dict = {"x": np.array([3.0])}
        assert_compiled_matches(expr, var_dict)

    def test_add_vectors(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        y = nvx.Variable((5,), name="y")
        expr = x + y
        var_dict = {
            "x": np.array([1, 2, 3, 4, 5.0]),
            "y": np.array([5, 4, 3, 2, 1.0]),
        }
        assert_compiled_matches(expr, var_dict)

    def test_sub(self):
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        y = nvx.Variable((3,), name="y")
        expr = x - y
        var_dict = {
            "x": np.array([1, 2, 3.0]),
            "y": np.array([3, 2, 1.0]),
        }
        assert_compiled_matches(expr, var_dict)

    def test_mul(self):
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr = x * 2.5
        var_dict = {"x": np.array([1, 2, 3.0])}
        assert_compiled_matches(expr, var_dict)

    def test_div(self):
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr = x / 2.0
        var_dict = {"x": np.array([2, 4, 6.0])}
        assert_compiled_matches(expr, var_dict)

    def test_pow(self):
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr = x ** 2
        var_dict = {"x": np.array([1, 2, 3.0])}
        assert_compiled_matches(expr, var_dict)

    def test_neg(self):
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr = -x
        var_dict = {"x": np.array([1, -2, 3.0])}
        assert_compiled_matches(expr, var_dict)


class TestMatrixOperations:
    """Test matrix operations."""

    def test_matmul(self):
        reset_variable_ids()
        A = nvx.Variable((3, 4), name="A")
        B = nvx.Variable((4, 2), name="B")
        expr = A @ B
        var_dict = {
            "A": np.random.randn(3, 4),
            "B": np.random.randn(4, 2),
        }
        assert_compiled_matches(expr, var_dict)

    def test_transpose(self):
        reset_variable_ids()
        A = nvx.Variable((3, 4), name="A")
        expr = A.T
        var_dict = {"A": np.random.randn(3, 4)}
        assert_compiled_matches(expr, var_dict)

    def test_flatten(self):
        reset_variable_ids()
        A = nvx.Variable((3, 4), name="A")
        expr = A.flatten()
        var_dict = {"A": np.random.randn(3, 4)}
        assert_compiled_matches(expr, var_dict)

    def test_complex_matrix_expr(self):
        reset_variable_ids()
        A = nvx.Variable((4, 4), name="A")
        B = nvx.Variable((4, 4), name="B")
        expr = A @ B.T + B @ A.T - A * B
        var_dict = {
            "A": np.random.randn(4, 4),
            "B": np.random.randn(4, 4),
        }
        assert_compiled_matches(expr, var_dict)


class TestIndexing:
    """Test indexing operations."""

    def test_integer_index(self):
        reset_variable_ids()
        A = nvx.Variable((5, 5), name="A")
        expr = A[0]
        var_dict = {"A": np.random.randn(5, 5)}
        assert_compiled_matches(expr, var_dict)

    def test_tuple_index(self):
        reset_variable_ids()
        A = nvx.Variable((5, 5), name="A")
        expr = A[1, 2]
        var_dict = {"A": np.random.randn(5, 5)}
        assert_compiled_matches(expr, var_dict)

    def test_slice_index(self):
        reset_variable_ids()
        A = nvx.Variable((5, 5), name="A")
        expr = A[1:3]
        var_dict = {"A": np.random.randn(5, 5)}
        assert_compiled_matches(expr, var_dict)

    def test_row_slice(self):
        reset_variable_ids()
        A = nvx.Variable((5, 5), name="A")
        expr = A[0, :]
        var_dict = {"A": np.random.randn(5, 5)}
        assert_compiled_matches(expr, var_dict)

    def test_col_slice(self):
        reset_variable_ids()
        A = nvx.Variable((5, 5), name="A")
        expr = A[:, 0]
        var_dict = {"A": np.random.randn(5, 5)}
        assert_compiled_matches(expr, var_dict)


class TestAtoms:
    """Test atom operations (norm, trace, etc.)."""

    def test_norm_l2(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        expr = nvx.norm(x)
        var_dict = {"x": np.random.randn(5)}
        assert_compiled_matches(expr, var_dict)

    def test_norm_l1(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        expr = nvx.norm(x, ord=1)
        var_dict = {"x": np.random.randn(5)}
        assert_compiled_matches(expr, var_dict)

    def test_norm_fro(self):
        reset_variable_ids()
        A = nvx.Variable((4, 4), name="A")
        expr = nvx.norm(A, ord="fro")
        var_dict = {"A": np.random.randn(4, 4)}
        assert_compiled_matches(expr, var_dict)

    def test_trace(self):
        reset_variable_ids()
        A = nvx.Variable((4, 4), name="A")
        expr = nvx.trace(A)
        var_dict = {"A": np.random.randn(4, 4)}
        assert_compiled_matches(expr, var_dict)

    def test_det(self):
        reset_variable_ids()
        A = nvx.Variable((3, 3), name="A")
        expr = nvx.det(A)
        var_dict = {"A": np.random.randn(3, 3)}
        assert_compiled_matches(expr, var_dict, rtol=1e-8, atol=1e-8)

    def test_sum(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        expr = nvx.sum(x)
        var_dict = {"x": np.random.randn(5)}
        assert_compiled_matches(expr, var_dict)

    def test_abs(self):
        reset_variable_ids()
        from nvxpy.atoms import abs as nvx_abs
        x = nvx.Variable((5,), name="x")
        expr = nvx_abs(x)
        var_dict = {"x": np.random.randn(5)}
        assert_compiled_matches(expr, var_dict)

    def test_amax(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        expr = nvx.amax(x)
        var_dict = {"x": np.random.randn(5)}
        assert_compiled_matches(expr, var_dict)

    def test_amin(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        expr = nvx.amin(x)
        var_dict = {"x": np.random.randn(5)}
        assert_compiled_matches(expr, var_dict)

    def test_maximum(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        y = nvx.Variable((5,), name="y")
        expr = nvx.maximum(x, y)
        var_dict = {
            "x": np.random.randn(5),
            "y": np.random.randn(5),
        }
        assert_compiled_matches(expr, var_dict)


class TestComplexExpressions:
    """Test complex nested expressions."""

    def test_nested_operations(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        y = nvx.Variable((5,), name="y")
        expr = (x + y) * (x - y) / 2 + x ** 2
        var_dict = {
            "x": np.random.randn(5),
            "y": np.random.randn(5),
        }
        assert_compiled_matches(expr, var_dict)

    def test_norm_of_expression(self):
        reset_variable_ids()
        x = nvx.Variable((5,), name="x")
        y = nvx.Variable((5,), name="y")
        expr = nvx.norm(x - y) ** 2
        var_dict = {
            "x": np.random.randn(5),
            "y": np.random.randn(5),
        }
        assert_compiled_matches(expr, var_dict)

    def test_matrix_expression_with_atoms(self):
        reset_variable_ids()
        A = nvx.Variable((4, 4), name="A")
        B = nvx.Variable((4, 4), name="B")
        expr = nvx.trace(A @ B.T) + nvx.norm(A - B, ord="fro") ** 2
        var_dict = {
            "A": np.random.randn(4, 4),
            "B": np.random.randn(4, 4),
        }
        assert_compiled_matches(expr, var_dict)

    def test_deep_nesting(self):
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        # Build deep expression: ((((x + 1) * 2) - 3) / 4) ** 0.5
        expr = x
        expr = expr + 1
        expr = expr * 2
        expr = expr - 3
        expr = expr / 4
        expr = expr ** 0.5
        var_dict = {"x": np.array([10.0, 20.0, 30.0])}  # Ensure positive after transforms
        assert_compiled_matches(expr, var_dict)

    def test_wide_expression(self):
        reset_variable_ids()
        # Many variables combined
        vars = [nvx.Variable((3,), name=f"x{i}") for i in range(10)]
        var_dict = {f"x{i}": np.random.randn(3) for i in range(10)}

        expr = vars[0]
        for v in vars[1:]:
            expr = expr + v

        assert_compiled_matches(expr, var_dict)

    def test_cse_opportunity(self):
        """Test that common subexpressions give correct results."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        y = nvx.Variable((3,), name="y")

        # (x + y) appears twice
        common = x + y
        expr = common * common + common

        var_dict = {
            "x": np.array([1, 2, 3.0]),
            "y": np.array([4, 5, 6.0]),
        }
        assert_compiled_matches(expr, var_dict)


class TestFunctions:
    """Test user-defined Function callables."""

    def test_simple_function(self):
        reset_variable_ids()

        def my_add(x, y):
            return x + y

        wrapped = nvx.Function(my_add)
        x = nvx.Variable((3,), name="x")
        y = nvx.Variable((3,), name="y")
        expr = wrapped(x, y)

        var_dict = {
            "x": np.array([1, 2, 3.0]),
            "y": np.array([4, 5, 6.0]),
        }
        assert_compiled_matches(expr, var_dict)

    def test_function_with_matrix_ops(self):
        reset_variable_ids()

        def mat_prod(A, B):
            return A @ B.T

        wrapped = nvx.Function(mat_prod)
        A = nvx.Variable((3, 3), name="A")
        B = nvx.Variable((3, 3), name="B")
        expr = wrapped(A, B) + A

        var_dict = {
            "A": np.random.randn(3, 3),
            "B": np.random.randn(3, 3),
        }
        assert_compiled_matches(expr, var_dict)

    def test_function_in_larger_expression(self):
        reset_variable_ids()

        def my_norm(x):
            return np.sqrt(np.sum(x ** 2))

        wrapped = nvx.Function(my_norm)
        x = nvx.Variable((5,), name="x")
        y = nvx.Variable((5,), name="y")
        expr = wrapped(x - y) ** 2 + wrapped(x + y)

        var_dict = {
            "x": np.random.randn(5),
            "y": np.random.randn(5),
        }
        assert_compiled_matches(expr, var_dict)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_variable(self):
        """Expression is just a variable."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        var_dict = {"x": np.array([1, 2, 3.0])}

        # Just the variable itself
        compiled = compile_expression(x)
        result = eval_compiled(compiled, var_dict)
        np.testing.assert_array_equal(result, var_dict["x"])

    def test_constant_expression(self):
        """Expression with a constant."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        const = np.array([10, 20, 30.0])
        expr = x + const
        var_dict = {"x": np.array([1, 2, 3.0])}
        assert_compiled_matches(expr, var_dict)

    def test_scalar_constant(self):
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr = x + 5
        var_dict = {"x": np.array([1, 2, 3.0])}
        assert_compiled_matches(expr, var_dict)

    def test_broadcasting(self):
        reset_variable_ids()
        A = nvx.Variable((3, 3), name="A")
        x = nvx.Variable((3,), name="x")
        # Matrix + vector (broadcasting)
        expr = A + x
        var_dict = {
            "A": np.random.randn(3, 3),
            "x": np.random.randn(3),
        }
        assert_compiled_matches(expr, var_dict)


class TestContainers:
    """Test container (dict, list, tuple) compilation."""

    def test_list_of_expressions(self):
        """Compile a list containing expressions."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        y = nvx.Variable((3,), name="y")
        expr_list = [x + 1, y * 2, x - y]
        var_dict = {
            "x": np.array([1, 2, 3.0]),
            "y": np.array([4, 5, 6.0]),
        }

        # Test IR compilation
        compiled = compile_expression(expr_list)
        ir_result = eval_compiled(compiled, var_dict)
        assert isinstance(ir_result, list)
        assert len(ir_result) == 3
        np.testing.assert_allclose(ir_result[0], np.array([2, 3, 4.0]))
        np.testing.assert_allclose(ir_result[1], np.array([8, 10, 12.0]))
        np.testing.assert_allclose(ir_result[2], np.array([-3, -3, -3.0]))

        # Test codegen compilation
        codegen_func = compile_to_function(expr_list)
        codegen_result = codegen_func(var_dict)
        assert isinstance(codegen_result, list)
        np.testing.assert_allclose(codegen_result[0], ir_result[0])
        np.testing.assert_allclose(codegen_result[1], ir_result[1])
        np.testing.assert_allclose(codegen_result[2], ir_result[2])

    def test_tuple_of_expressions(self):
        """Compile a tuple containing expressions."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr_tuple = (x + 1, x * 2)
        var_dict = {"x": np.array([1, 2, 3.0])}

        # Test IR compilation
        compiled = compile_expression(expr_tuple)
        ir_result = eval_compiled(compiled, var_dict)
        assert isinstance(ir_result, tuple)
        assert len(ir_result) == 2
        np.testing.assert_allclose(ir_result[0], np.array([2, 3, 4.0]))
        np.testing.assert_allclose(ir_result[1], np.array([2, 4, 6.0]))

        # Test codegen compilation
        codegen_func = compile_to_function(expr_tuple)
        codegen_result = codegen_func(var_dict)
        assert isinstance(codegen_result, tuple)
        np.testing.assert_allclose(codegen_result[0], ir_result[0])
        np.testing.assert_allclose(codegen_result[1], ir_result[1])

    def test_dict_of_expressions(self):
        """Compile a dict containing expressions."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr_dict = {"squared": x ** 2, "doubled": x * 2, "negated": -x}
        var_dict = {"x": np.array([1, 2, 3.0])}

        # Test IR compilation
        compiled = compile_expression(expr_dict)
        ir_result = eval_compiled(compiled, var_dict)
        assert isinstance(ir_result, dict)
        assert set(ir_result.keys()) == {"squared", "doubled", "negated"}
        np.testing.assert_allclose(ir_result["squared"], np.array([1, 4, 9.0]))
        np.testing.assert_allclose(ir_result["doubled"], np.array([2, 4, 6.0]))
        np.testing.assert_allclose(ir_result["negated"], np.array([-1, -2, -3.0]))

        # Test codegen compilation
        codegen_func = compile_to_function(expr_dict)
        codegen_result = codegen_func(var_dict)
        assert isinstance(codegen_result, dict)
        np.testing.assert_allclose(codegen_result["squared"], ir_result["squared"])
        np.testing.assert_allclose(codegen_result["doubled"], ir_result["doubled"])
        np.testing.assert_allclose(codegen_result["negated"], ir_result["negated"])

    def test_mixed_container(self):
        """Compile container with mix of expressions and constants."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr_list = [x + 1, np.array([10, 20, 30.0]), x * 2]
        var_dict = {"x": np.array([1, 2, 3.0])}

        # Test IR compilation
        compiled = compile_expression(expr_list)
        ir_result = eval_compiled(compiled, var_dict)
        assert isinstance(ir_result, list)
        np.testing.assert_allclose(ir_result[0], np.array([2, 3, 4.0]))
        np.testing.assert_allclose(ir_result[1], np.array([10, 20, 30.0]))
        np.testing.assert_allclose(ir_result[2], np.array([2, 4, 6.0]))


class TestGeneratedCode:
    """Test properties of generated code."""

    def test_source_available(self):
        """Check that source code is available for debugging."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        expr = x + 1
        func = compile_to_function(expr)
        assert hasattr(func, '_source')
        assert 'def _compiled_eval' in func._source

    def test_var_names_available(self):
        """Check that variable names are available."""
        reset_variable_ids()
        x = nvx.Variable((3,), name="x")
        y = nvx.Variable((3,), name="y")
        expr = x + y
        func = compile_to_function(expr)
        assert hasattr(func, '_var_names')
        assert set(func._var_names) == {'x', 'y'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
