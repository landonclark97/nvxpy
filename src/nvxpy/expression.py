import autograd.numpy as np

from .constraint import Constraint
from .set import Set
from .constants import Curvature as C


# Type alias for values that can be used with expressions
ExprLike = "BaseExpr | np.ndarray | int | float | complex"


def broadcast_shapes(left_shape: tuple, right_shape: tuple) -> tuple:
    """Compute the result shape of broadcasting two shapes together (NumPy rules)."""
    if not left_shape or not right_shape:
        return left_shape if not right_shape else right_shape

    left_shape = list(left_shape)
    right_shape = list(right_shape)

    while len(left_shape) < len(right_shape):
        left_shape.insert(0, 1)
    while len(right_shape) < len(left_shape):
        right_shape.insert(0, 1)

    result_shape = []
    for left_dim, right_dim in zip(left_shape, right_shape):
        if left_dim == 1 or right_dim == 1:
            result_shape.append(max(left_dim, right_dim))
        elif left_dim == right_dim:
            result_shape.append(left_dim)
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {tuple(left_shape)}, {tuple(right_shape)}")

    return tuple(result_shape)


def expr_to_str(expr: ExprLike) -> str:
    """Convert an expression-like object to a string representation."""
    if isinstance(expr, BaseExpr):
        return str(expr)
    elif isinstance(expr, np.ndarray):
        return f"Const(shape={expr.shape}, id={id(expr)})"
    elif np.isscalar(expr):
        return f"Const(type={type(expr).__name__}, id={id(expr)})"
    return f"Unknown({type(expr).__name__})"


class BaseExpr:
    """Base class for all expression types in nvxpy.

    Provides arithmetic operators (+, -, *, /, @, **) that build expression trees,
    comparison operators (>=, <=, ==) that create constraints, and matrix operators
    (>>, <<) for semidefinite constraints.

    The ^ operator creates discrete set membership constraints when used with a
    DiscreteSet or list of values.
    """

    __array_priority__ = 100

    @property
    def T(self):
        return Expr("transpose", self)
    
    def flatten(self):
        return Expr("flatten", self)

    def __add__(self, other):
        return Expr("add", self, other)

    def __radd__(self, other):
        return Expr("add", other, self)

    def __sub__(self, other):
        return Expr("sub", self, other)

    def __rsub__(self, other):
        return Expr("sub", other, self)

    def __mul__(self, other):
        return Expr("mul", self, other)

    def __rmul__(self, other):
        return Expr("mul", other, self)

    def __matmul__(self, other):
        return Expr("matmul", self, other)

    def __rmatmul__(self, other):
        return Expr("matmul", other, self)

    def __truediv__(self, other):
        return Expr("div", self, other)
    
    def __rtruediv__(self, other):
        return Expr("div", other, self)

    def __pow__(self, other):
        return Expr("pow", self, other)

    def __neg__(self):
        return Expr("neg", self)

    def __getitem__(self, key):
        return Expr("getitem", self, key)

    def __ge__(self, other):
        return Constraint(self, ">=", other)

    def __le__(self, other):
        return Constraint(self, "<=", other)

    def __eq__(self, other):
        return Constraint(self, "==", other)

    def __rshift__(self, other):
        return Constraint(self, ">>", other)

    def __lshift__(self, other):
        return Constraint(self, "<<", other)

    def __xor__(self, other):
        if isinstance(other, (list, tuple)):
            from .sets.discrete_set import _coerce_to_discrete_set
            other = _coerce_to_discrete_set(other)
        if not isinstance(other, Set):
            raise TypeError("Right operand of ^ must be a Set object")
        return other.constrain(self)


class Expr(BaseExpr):
    """An expression node in the computation graph.

    Represents an operation applied to one or two operands. Operations include
    arithmetic (add, sub, mul, div, pow, neg), matrix operations (matmul, transpose),
    and indexing (getitem).

    Attributes:
        op: The operation name (e.g., "add", "mul", "matmul")
        left: The left operand (or only operand for unary operations)
        right: The right operand (None for unary operations)
    """

    def __init__(self, op: str, left, right=None):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        left_str = expr_to_str(self.left)
        right_str = None
        if self.right is not None:
            right_str = expr_to_str(self.right)
        return f"{self.op}({left_str}, {right_str})" if right_str else f"{self.op}({left_str})"
    
    def __hash__(self):
        return hash(str(self))

    @property
    def value(self):
        from .parser import eval_expression
        return eval_expression(self, None, use_value=True)
    
    @property
    def curvature(self):
        from .variable import Variable

        def negate_curvature(conv):
            if conv == C.CONVEX:
                return C.CONCAVE
            elif conv == C.CONCAVE:
                return C.CONVEX
            else:
                return conv
            
        if isinstance(self.left, Expr) or isinstance(self.left, Variable):
            left = self.left.curvature
            if left == C.CONSTANT:
                left_val = self.left.value
        else:
            left = C.CONSTANT
            left_val = self.left
        if isinstance(self.right, Expr) or isinstance(self.right, Variable):
            right = self.right.curvature
            if right == C.CONSTANT:
                right_val = self.right.value
        elif self.right is not None:
            right = C.CONSTANT
            right_val = self.right
        else:
            right = None
            right_val = None

        if self.op in ("add", "sub"):
            if self.op == "sub":
                right = negate_curvature(right)
            if left == right:
                return left
            elif C.CONSTANT in (left, right):
                return left if right == C.CONSTANT else right
            elif C.AFFINE in (left, right):
                return left if right == C.AFFINE else right
            else:
                return C.UNKNOWN

        elif self.op == "mul":
            if left == C.CONSTANT:
                if all([left_val >= 0]):
                    return right
                elif all([left_val < 0]):
                    return negate_curvature(right)
            elif right == C.CONSTANT:
                if all([right_val >= 0]):
                    return left
                elif all([right_val < 0]):
                    return negate_curvature(left)
            return C.UNKNOWN

        elif self.op == "div":
            if right == C.CONSTANT and all([right_val >= 0]):
                return left
            elif right == C.CONSTANT and all([right_val < 0]):
                return negate_curvature(left)
            else:
                return C.UNKNOWN

        elif self.op == "pow":
            if right == C.CONSTANT:
                if right_val == 1:
                    return left
                elif right_val > 1 and left in (C.CONVEX, C.AFFINE):
                    return C.CONVEX
                elif 0 < right_val < 1 and left == C.CONCAVE:
                    return C.CONCAVE
                else:
                    return C.UNKNOWN
            else:
                return C.UNKNOWN

        elif self.op == "neg":
            return negate_curvature(left)

        elif self.op == "matmul":
            if left == C.AFFINE and right == C.CONSTANT:
                return C.AFFINE
            elif left == C.CONSTANT and right == C.AFFINE:
                return C.AFFINE
            else:
                return C.UNKNOWN

        elif self.op == "getitem":
            return left

        elif self.op == "transpose":
            return left
        
        elif self.op == "flatten":
            return left
        
        raise NotImplementedError(f"Curvature of Expr: {self.op} is not implemented")

    @property
    def shape(self):
        from .variable import Variable

        def get_shape(obj):
            if isinstance(obj, (Expr, Variable)):
                return obj.shape
            elif isinstance(obj, np.ndarray):
                return obj.shape
            elif np.isscalar(obj):
                return ()
            elif isinstance(obj, (slice, tuple)):
                return None
            else:
                raise ValueError(f"Cannot determine shape of {type(obj)}")

        left_shape = get_shape(self.left)
        right_shape = get_shape(self.right) if self.right is not None else None

        if self.op in ("add", "sub", "mul", "div"):
            return broadcast_shapes(left_shape, right_shape)

        elif self.op == "pow":
            return left_shape

        elif self.op == "neg":
            return left_shape

        elif self.op == "matmul":
            if len(left_shape) == 1 and len(right_shape) == 1:
                if left_shape[0] != right_shape[0]:
                    raise ValueError(f"Incompatible shapes for matmul: {left_shape}, {right_shape}")
                return ()
            elif len(left_shape) == 1:
                left_shape = (1,) + left_shape
            elif len(right_shape) == 1:
                right_shape = right_shape + (1,)
            
            if left_shape[-1] != right_shape[-2]:
                raise ValueError(f"Incompatible shapes for matmul: {left_shape}, {right_shape}")
            
            left_batch = left_shape[:-2]
            right_batch = right_shape[:-2]
            
            max_batch_dims = max(len(left_batch), len(right_batch))
            left_batch = (1,) * (max_batch_dims - len(left_batch)) + left_batch
            right_batch = (1,) * (max_batch_dims - len(right_batch)) + right_batch
            
            batch_shape = []
            for left_dim, right_dim in zip(left_batch, right_batch):
                if left_dim == 1 or right_dim == 1:
                    batch_shape.append(max(left_dim, right_dim))
                elif left_dim == right_dim:
                    batch_shape.append(left_dim)
                else:
                    raise ValueError(f"Incompatible batch dimensions for matmul: {left_shape}, {right_shape}")
            
            result_shape = tuple(batch_shape) + (left_shape[-2], right_shape[-1])
            
            if len(left_shape) == 1:
                result_shape = result_shape[1:]
            
            return result_shape

        elif self.op == "getitem":
            if isinstance(self.right, tuple):
                new_shape = []
                orig_shape = list(left_shape)
                dim_idx = 0
                for idx in self.right:
                    if dim_idx >= len(orig_shape):
                        raise IndexError(f"too many indices for array of dimension {len(orig_shape)}")
                    dim_size = orig_shape[dim_idx]
                    if isinstance(idx, slice):
                        start = 0 if idx.start is None else idx.start
                        stop = dim_size if idx.stop is None else idx.stop
                        step = 1 if idx.step is None else idx.step
                        if step == 0:
                            raise ValueError("Slice step cannot be zero")
                        # Normalize negative indices
                        if start < 0:
                            start = max(0, dim_size + start)
                        if stop < 0:
                            stop = max(0, dim_size + stop)
                        # Clamp to valid range
                        start = min(start, dim_size)
                        stop = min(stop, dim_size)
                        new_shape.append(max(0, (stop - start + step - 1) // step) if step > 0 else max(0, (start - stop - step - 1) // (-step)))
                        dim_idx += 1
                    elif isinstance(idx, int):
                        # Validate index bounds
                        if idx < -dim_size or idx >= dim_size:
                            raise IndexError(f"index {idx} is out of bounds for axis {dim_idx} with size {dim_size}")
                        dim_idx += 1
                        continue
                    else:
                        raise ValueError(f"Unsupported index type: {type(idx)}")
                return tuple(new_shape) if new_shape else (1,)
            elif isinstance(self.right, (int, slice)):
                if isinstance(self.right, int):
                    dim_size = left_shape[0] if left_shape else 0
                    if self.right < -dim_size or self.right >= dim_size:
                        raise IndexError(f"index {self.right} is out of bounds for axis 0 with size {dim_size}")
                    return left_shape[1:] if len(left_shape) > 1 else (1,)
                else:
                    dim_size = left_shape[0] if left_shape else 0
                    start = 0 if self.right.start is None else self.right.start
                    stop = dim_size if self.right.stop is None else self.right.stop
                    step = 1 if self.right.step is None else self.right.step
                    if step == 0:
                        raise ValueError("Slice step cannot be zero")
                    # Normalize negative indices
                    if start < 0:
                        start = max(0, dim_size + start)
                    if stop < 0:
                        stop = max(0, dim_size + stop)
                    # Clamp to valid range
                    start = min(start, dim_size)
                    stop = min(stop, dim_size)
                    slice_len = max(0, (stop - start + step - 1) // step) if step > 0 else max(0, (start - stop - step - 1) // (-step))
                    return (slice_len,) + left_shape[1:]
            else:
                raise ValueError(f"Unsupported index type: {type(self.right)}")

        elif self.op == "transpose":
            return left_shape[::-1]
        
        elif self.op == "flatten":
            return (np.prod(left_shape),)
        
        raise NotImplementedError(f"Shape inference for operation {self.op} is not implemented")
