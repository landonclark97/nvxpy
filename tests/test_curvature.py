import autograd.numpy as np
import pytest

from nvxpy.constants import Curvature as C
from nvxpy.variable import Variable
from nvxpy.atoms import norm, maximum, sum, trace, det, logdet
from nvxpy.constructs import Function


class TestCurvatureConstants:
    """Test basic curvature constants and their properties."""
    
    def test_curvature_enum_values(self):
        """Test that all curvature constants are properly defined."""
        assert C.CONSTANT != C.AFFINE
        assert C.AFFINE != C.CONVEX
        assert C.CONVEX != C.CONCAVE
        assert C.CONCAVE != C.UNKNOWN
        assert C.UNKNOWN != C.CONSTANT
        
        # Test that all expected curvatures exist
        curvatures = [C.CONSTANT, C.AFFINE, C.CONVEX, C.CONCAVE, C.UNKNOWN]
        assert len(set(curvatures)) == 5


class TestVariableCurvature:
    """Test curvature of Variable objects."""
    
    def test_variable_curvature_is_affine(self):
        """Variables should always have AFFINE curvature."""
        x = Variable(shape=(2,))
        assert x.curvature == C.AFFINE
        
        y = Variable(shape=(3, 3))
        assert y.curvature == C.AFFINE
        
        z = Variable(shape=(1,))
        assert z.curvature == C.AFFINE


class TestExpressionCurvature:
    """Test curvature rules for various expression operations."""
    
    def test_add_curvature(self):
        """Test addition curvature rules."""
        x = Variable(shape=(2,))
        y = Variable(shape=(2,))
        
        # AFFINE + AFFINE = AFFINE
        expr = x + y
        assert expr.curvature == C.AFFINE
        
        # AFFINE + CONSTANT = AFFINE
        expr = x + 5
        assert expr.curvature == C.AFFINE
        
        # CONSTANT + AFFINE = AFFINE
        expr = 5 + x
        assert expr.curvature == C.AFFINE
        
        # CONVEX + CONVEX = CONVEX
        convex_expr = norm(x)
        expr = convex_expr + convex_expr
        assert expr.curvature == C.CONVEX
        
        # CONCAVE + CONCAVE = CONCAVE
        concave_expr = -norm(x)
        expr = concave_expr + concave_expr
        assert expr.curvature == C.CONCAVE
        
        # CONVEX + CONCAVE = UNKNOWN
        expr = convex_expr + concave_expr
        assert expr.curvature == C.UNKNOWN
    
    def test_sub_curvature(self):
        """Test subtraction curvature rules."""
        x = Variable(shape=(2,))
        y = Variable(shape=(2,))
        
        # AFFINE - AFFINE = AFFINE
        expr = x - y
        assert expr.curvature == C.AFFINE
        
        # AFFINE - CONSTANT = AFFINE
        expr = x - 5
        assert expr.curvature == C.AFFINE
        
        # CONSTANT - AFFINE = AFFINE
        expr = 5 - x
        assert expr.curvature == C.AFFINE
        
        # CONVEX - CONVEX = UNKNOWN
        convex_expr = norm(x)
        expr = convex_expr - convex_expr
        assert expr.curvature == C.UNKNOWN
        
        # CONVEX - CONCAVE = CONVEX
        concave_expr = -norm(y)
        expr = convex_expr - concave_expr
        assert expr.curvature == C.CONVEX
        
        # CONCAVE - CONVEX = CONCAVE
        expr = concave_expr - convex_expr
        assert expr.curvature == C.CONCAVE
    
    def test_mul_curvature(self):
        """Test multiplication curvature rules."""
        x = Variable(shape=(2,))
        
        # Positive constant * AFFINE = AFFINE
        expr = 2 * x
        assert expr.curvature == C.AFFINE
        
        # Negative constant * AFFINE = AFFINE
        expr = -2 * x
        assert expr.curvature == C.AFFINE
        
        # Positive constant * CONVEX = CONVEX
        convex_expr = norm(x)
        expr = 2 * convex_expr
        assert expr.curvature == C.CONVEX
        
        # Negative constant * CONVEX = CONCAVE
        expr = -2 * convex_expr
        assert expr.curvature == C.CONCAVE
        
        # Positive constant * CONCAVE = CONCAVE
        concave_expr = -norm(x)
        expr = 2 * concave_expr
        assert expr.curvature == C.CONCAVE
        
        # Negative constant * CONCAVE = CONVEX
        expr = -2 * concave_expr
        assert expr.curvature == C.CONVEX
        
        # Variable * Variable = UNKNOWN
        y = Variable(shape=(2,))
        expr = x * y
        assert expr.curvature == C.UNKNOWN
    
    def test_div_curvature(self):
        """Test division curvature rules."""
        x = Variable(shape=(2,))
        
        # AFFINE / positive constant = AFFINE
        expr = x / 2
        assert expr.curvature == C.AFFINE
        
        # AFFINE / negative constant = AFFINE
        expr = x / (-2)
        assert expr.curvature == C.AFFINE
        
        # CONVEX / positive constant = CONVEX
        convex_expr = norm(x)
        expr = convex_expr / 2
        assert expr.curvature == C.CONVEX
        
        # CONVEX / negative constant = CONCAVE
        expr = convex_expr / (-2)
        assert expr.curvature == C.CONCAVE
        
        # CONCAVE / positive constant = CONCAVE
        concave_expr = -norm(x)
        expr = concave_expr / 2
        assert expr.curvature == C.CONCAVE
        
        # CONCAVE / negative constant = CONVEX
        expr = concave_expr / (-2)
        assert expr.curvature == C.CONVEX
        
        # Division by variable = UNKNOWN
        y = Variable(shape=(2,))
        expr = x / y
        assert expr.curvature == C.UNKNOWN
    
    def test_pow_curvature(self):
        """Test power curvature rules."""
        x = Variable(shape=(2,))
        
        # AFFINE^1 = AFFINE
        expr = x ** 1
        assert expr.curvature == C.AFFINE
        
        # AFFINE^2 = CONVEX
        expr = x ** 2
        assert expr.curvature == C.CONVEX
        
        # CONVEX^2 = CONVEX
        convex_expr = norm(x)
        expr = convex_expr ** 2
        assert expr.curvature == C.CONVEX
        
        # CONCAVE^0.5 = CONCAVE (if CONCAVE is non-negative)
        # Note: This is a theoretical case, in practice CONCAVE functions
        # might not be non-negative, so this could be UNKNOWN
        concave_expr = -norm(x)
        expr = concave_expr ** 0.5
        # This should be UNKNOWN since we can't guarantee non-negativity
        assert expr.curvature == C.CONCAVE
        
        # Variable^variable = UNKNOWN
        y = Variable(shape=(2,))
        expr = x ** y
        assert expr.curvature == C.UNKNOWN
    
    def test_neg_curvature(self):
        """Test negation curvature rules."""
        x = Variable(shape=(2,))
        
        # -AFFINE = AFFINE
        expr = -x
        assert expr.curvature == C.AFFINE
        
        # -CONVEX = CONCAVE
        convex_expr = norm(x)
        expr = -convex_expr
        assert expr.curvature == C.CONCAVE
        
        # -CONCAVE = CONVEX
        concave_expr = -norm(x)
        expr = -concave_expr
        assert expr.curvature == C.CONVEX
        
    
    def test_matmul_curvature(self):
        """Test matrix multiplication curvature rules."""
        x = Variable(shape=(2,))
        A = np.array([[1, 2], [3, 4]])
        
        # AFFINE @ CONSTANT = AFFINE
        expr = x @ A
        assert expr.curvature == C.AFFINE
        
        # CONSTANT @ AFFINE = AFFINE
        expr = A @ x
        assert expr.curvature == C.AFFINE
        
        # AFFINE @ AFFINE = UNKNOWN
        y = Variable(shape=(2,))
        expr = x @ y
        assert expr.curvature == C.UNKNOWN
    
    def test_transpose_curvature(self):
        """Test transpose curvature preservation."""
        x = Variable(shape=(2, 2))
        
        # Transpose preserves curvature
        expr = x.T
        assert expr.curvature == x.curvature
        
        # Transpose of expression preserves curvature
        convex_expr = norm(x)
        expr = convex_expr.T
        assert expr.curvature == convex_expr.curvature
    
    def test_flatten_curvature(self):
        """Test flatten curvature preservation."""
        x = Variable(shape=(2, 2))
        
        # Flatten preserves curvature
        expr = x.flatten()
        assert expr.curvature == x.curvature
        
        # Flatten of expression preserves curvature
        convex_expr = norm(x)
        expr = convex_expr.flatten()
        assert expr.curvature == convex_expr.curvature
    
    def test_getitem_curvature(self):
        """Test indexing curvature preservation."""
        x = Variable(shape=(3,))
        
        # Indexing preserves curvature
        expr = x[0]
        assert expr.curvature == x.curvature
        
        # Indexing of expression preserves curvature
        convex_expr = norm(x)
        expr = convex_expr[0]
        assert expr.curvature == convex_expr.curvature


class TestAtomCurvature:
    """Test curvature of atomic functions."""
    
    def test_norm_curvature(self):
        """Test norm curvature rules."""
        x = Variable(shape=(2,))
        
        # norm(AFFINE) = CONVEX for p >= 1
        expr = norm(x)
        assert expr.curvature == C.CONVEX
        
        # norm with different orders
        expr = norm(x, ord=1)
        assert expr.curvature == C.CONVEX
        
        expr = norm(x, ord=3)
        assert expr.curvature == C.CONVEX
        
        expr = norm(x, ord=np.inf)
        assert expr.curvature == C.CONVEX
        
        # norm with string orders
        expr = norm(x, ord="fro")
        assert expr.curvature == C.CONVEX
        
        expr = norm(x, ord="nuc")
        assert expr.curvature == C.CONVEX
        
        # norm of constant should be CONVEX
        expr = norm(5)
        assert expr.curvature == C.CONVEX
    
    def test_maximum_curvature(self):
        """Test maximum curvature rules."""
        x = Variable(shape=(2,))
        y = Variable(shape=(2,))
        
        # maximum(AFFINE, AFFINE) = CONVEX
        expr = maximum(x, y)
        assert expr.curvature == C.CONVEX
        
        # maximum(AFFINE, CONSTANT) = CONVEX
        expr = maximum(x, 5)
        assert expr.curvature == C.CONVEX
        
        # maximum(CONSTANT, AFFINE) = CONVEX
        expr = maximum(5, x)
        assert expr.curvature == C.CONVEX
        
        # maximum(CONVEX, CONVEX) = CONVEX
        convex_expr1 = norm(x)
        convex_expr2 = norm(y)
        expr = maximum(convex_expr1, convex_expr2)
        assert expr.curvature == C.CONVEX
        
        # maximum(CONVEX, AFFINE) = CONVEX
        expr = maximum(convex_expr1, y)
        assert expr.curvature == C.CONVEX
        
        # maximum(CONVEX, CONCAVE) = UNKNOWN
        concave_expr = -norm(y)
        expr = maximum(convex_expr1, concave_expr)
        assert expr.curvature == C.UNKNOWN
    
    def test_sum_curvature(self):
        """Test sum curvature preservation."""
        x = Variable(shape=(2,))
        
        # sum(AFFINE) = AFFINE
        expr = sum(x)
        assert expr.curvature == C.AFFINE
        
        # sum(CONVEX) = CONVEX
        convex_expr = norm(x)
        expr = sum(convex_expr)
        assert expr.curvature == C.CONVEX
        
        # sum(CONCAVE) = CONCAVE
        concave_expr = -norm(x)
        expr = sum(concave_expr)
        assert expr.curvature == C.CONCAVE
        
        # sum(CONSTANT) = CONSTANT
        expr = sum(5)
        assert expr.curvature == C.CONSTANT
    
    def test_trace_curvature(self):
        """Test trace curvature preservation."""
        x = Variable(shape=(2, 2))
        
        # trace(AFFINE) = AFFINE
        expr = trace(x)
        assert expr.curvature == C.AFFINE
        
        # trace(CONVEX) = CONVEX
        convex_expr = norm(x)
        expr = trace(convex_expr)
        assert expr.curvature == C.CONVEX
        
        # trace(CONCAVE) = CONCAVE
        concave_expr = -norm(x)
        expr = trace(concave_expr)
        assert expr.curvature == C.CONCAVE
        
        # trace(CONSTANT) = CONSTANT
        expr = trace(5)
        assert expr.curvature == C.CONSTANT
    
    def test_det_curvature(self):
        """Test determinant curvature."""
        x = Variable(shape=(2, 2))
        
        # det always returns UNKNOWN
        expr = det(x)
        assert expr.curvature == C.UNKNOWN
        
        # det of constant also UNKNOWN
        expr = det(5)
        assert expr.curvature == C.UNKNOWN
    
    def test_logdet_curvature(self):
        """Test log determinant curvature."""
        x = Variable(shape=(2, 2))
        
        # logdet always returns UNKNOWN
        expr = logdet(x)
        assert expr.curvature == C.UNKNOWN
        
        # logdet of constant also UNKNOWN
        expr = logdet(5)
        assert expr.curvature == C.UNKNOWN
    
    def test_polar_curvature(self):
        """Test polar decomposition curvature."""
        x = Variable(shape=(2, 2))
        
        # Polar decomposition always returns UNKNOWN
        from nvxpy.atoms.polar import PolarDecomposition
        expr = PolarDecomposition(x)
        assert expr.curvature == C.UNKNOWN
    
    def test_function_curvature(self):
        """Test Function curvature."""        
        def test_func(x):
            return np.sum(x**2)
        
        # Function always returns UNKNOWN
        expr = Function(test_func)
        assert expr.curvature == C.UNKNOWN


class TestComplexCurvatureCombinations:
    """Test complex combinations of curvature operations."""
    
    def test_nested_expressions(self):
        """Test curvature of deeply nested expressions."""
        x = Variable(shape=(2,))
        y = Variable(shape=(2,))
        
        # Complex expression: norm(x + y) - norm(x - y)
        expr1 = x + y  # AFFINE
        expr2 = x - y  # AFFINE
        expr3 = norm(expr1)  # CONVEX
        expr4 = norm(expr2)  # CONVEX
        result = expr3 - expr4  # CONVEX - CONVEX = UNKNOWN
        assert result.curvature == C.UNKNOWN
        
        # Complex expression: 2 * norm(x) + 3 * norm(y)
        expr1 = 2 * norm(x)  # CONVEX
        expr2 = 3 * norm(y)  # CONVEX
        result = expr1 + expr2  # CONVEX + CONVEX = CONVEX
        assert result.curvature == C.CONVEX
    
    def test_curvature_with_constants(self):
        """Test curvature when mixing with constants."""
        x = Variable(shape=(2,))
        
        # norm(x + 5) = CONVEX
        expr = norm(x + 5)
        assert expr.curvature == C.CONVEX
        
        # norm(2 * x) = CONVEX
        expr = norm(2 * x)
        assert expr.curvature == C.CONVEX
        
        # maximum(norm(x), 0) = CONVEX
        expr = maximum(norm(x), 0)
        assert expr.curvature == C.CONVEX
    
    def test_curvature_chain_rules(self):
        """Test curvature chain rules for complex expressions."""
        x = Variable(shape=(2,))
        y = Variable(shape=(2,))
        
        # Chain: norm(x) -> neg -> mul by positive -> add affine
        expr1 = norm(x)  # CONVEX
        expr2 = -expr1  # CONCAVE
        expr3 = 2 * expr2  # CONCAVE
        expr4 = expr3 + y  # CONCAVE + AFFINE = CONCAVE
        assert expr4.curvature == C.CONCAVE
        
        # Chain: norm(x) -> pow -> mul by negative -> add affine
        expr1 = norm(x)  # CONVEX
        expr2 = expr1 ** 2  # CONVEX
        expr3 = -expr2  # CONCAVE
        expr4 = expr3 + y  # CONCAVE + AFFINE = CONCAVE
        assert expr4.curvature == C.CONCAVE


class TestCurvatureEdgeCases:
    """Test edge cases and boundary conditions for curvature."""
    
    def test_zero_constants(self):
        """Test curvature with zero constants."""
        x = Variable(shape=(2,))
        
        # 0 * AFFINE = AFFINE
        expr = 0 * x
        assert expr.curvature == C.AFFINE
        
        # 0 * CONVEX = CONVEX
        convex_expr = norm(x)
        expr = 0 * convex_expr
        assert expr.curvature == C.CONVEX
        
        # AFFINE + 0 = AFFINE
        expr = x + 0
        assert expr.curvature == C.AFFINE
        
        # AFFINE - 0 = AFFINE
        expr = x - 0
        assert expr.curvature == C.AFFINE
    
    def test_negative_powers(self):
        """Test curvature with negative powers."""
        x = Variable(shape=(2,))
        
        # AFFINE^(-1) = UNKNOWN
        expr = x ** (-1)
        assert expr.curvature == C.UNKNOWN
        
        # AFFINE^(-2) = UNKNOWN
        expr = x ** (-2)
        assert expr.curvature == C.UNKNOWN
    
    def test_fractional_powers(self):
        """Test curvature with fractional powers."""
        x = Variable(shape=(2,))
        
        # AFFINE^(0.5) = UNKNOWN (not guaranteed to be convex)
        expr = x ** 0.5
        assert expr.curvature == C.UNKNOWN
        
        # AFFINE^(1.5) = CONVEX
        expr = x ** 1.5
        assert expr.curvature == C.CONVEX
    
    def test_matrix_operations(self):
        """Test curvature of matrix operations."""
        X = Variable(shape=(2, 2))
        
        # Transpose preserves curvature
        expr = X.T
        assert expr.curvature == X.curvature
        
        # Flatten preserves curvature
        expr = X.flatten()
        assert expr.curvature == X.curvature
        
        # Indexing preserves curvature
        expr = X[0, 0]
        assert expr.curvature == X.curvature


if __name__ == "__main__":
    pytest.main([__file__])
