"""
Expression Compiler for nvxpy

This module provides a compiled evaluation approach for expression trees.
Instead of recursively interpreting the tree at each evaluation, we compile
the expression once into a sequence of operations (linear IR) that can be
executed efficiently.

The compilation process:
1. Traverse the expression tree once
2. Emit operations in topological order
3. Store constants and operation metadata
4. Return a compiled function that executes the operations sequentially

This eliminates:
- Recursive function call overhead
- Repeated isinstance() checks
- String comparisons for operation dispatch
"""

from typing import Callable, Dict, List, Tuple, Any, Optional, Iterable
from dataclasses import dataclass, field
import autograd.numpy as np

from .expression import Expr
from .variable import Variable
from .constructs.function import Function


@dataclass
class CompiledOp:
    """A single compiled operation."""
    op: str                          # Operation name
    result_idx: int                  # Index to store result
    left_idx: int                    # Index of left operand (-1 for constants)
    right_idx: int                   # Index of right operand (-1 for None/constants)
    left_const: Any = None           # Left constant value (if left_idx == -1)
    right_const: Any = None          # Right constant value (if right_idx == -1)
    atom_callable: Callable = None   # For atom operations (norm, trace, etc.)
    func_callable: Callable = None   # For Function operations
    func_kwargs: Dict = None         # Kwargs for Function
    getitem_key: Any = None          # For getitem operations


@dataclass
class CompiledExpression:
    """A compiled expression ready for fast evaluation."""
    ops: List[CompiledOp]            # Sequence of operations
    var_indices: Dict[str, int]      # Map variable names to slot indices
    result_idx: int                  # Index of final result
    num_slots: int                   # Total number of temporary slots needed

    # Pre-allocated workspace (created on first eval, reused)
    _workspace: Optional[List] = field(default=None, repr=False)

    def __call__(self, var_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the compiled expression with given variable values."""
        return eval_compiled(self, var_dict)

    def eval_with_values(self) -> np.ndarray:
        """Evaluate using Variable._value attributes."""
        return eval_compiled_with_values(self)


# Operation dispatch table - maps op names to functions
# Using a dict avoids if-elif chains
_BINARY_OPS = {
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "mul": lambda left, right: left * right,
    "div": lambda left, right: left / right,
    "pow": lambda left, right: left ** right,
    "matmul": lambda left, right: left @ right,
}

_UNARY_OPS = {
    "neg": lambda val: -val,
    "transpose": lambda val: val.T,
    "flatten": lambda val: val.flatten(),
}


class ExpressionCompiler:
    """Compiles expression trees into linear operation sequences."""

    def __init__(self):
        self.ops: List[CompiledOp] = []
        self.slot_counter = 0
        self.expr_to_slot: Dict[int, int] = {}  # expr id -> slot index
        self.var_indices: Dict[str, int] = {}    # var name -> slot index
        self.var_objects: Dict[str, Variable] = {}  # var name -> Variable object

    def _alloc_slot(self) -> int:
        """Allocate a new slot for intermediate results."""
        idx = self.slot_counter
        self.slot_counter += 1
        return idx

    def compile(self, expr) -> CompiledExpression:
        """Compile an expression tree into a CompiledExpression."""
        self.ops = []
        self.slot_counter = 0
        self.expr_to_slot = {}
        self.var_indices = {}
        self.var_objects = {}

        result_idx = self._compile_node(expr)

        return CompiledExpression(
            ops=self.ops,
            var_indices=self.var_indices,
            result_idx=result_idx,
            num_slots=self.slot_counter,
        )

    def _compile_node(self, expr) -> int:
        """
        Compile a single node and return its slot index.
        Returns -1 for constants (stored inline in the op).
        """
        # Check if already compiled (enables CSE)
        expr_id = id(expr)
        if expr_id in self.expr_to_slot:
            return self.expr_to_slot[expr_id]

        if isinstance(expr, Variable):
            # Variables get their own slots, filled at eval time
            if expr.name not in self.var_indices:
                slot = self._alloc_slot()
                self.var_indices[expr.name] = slot
                self.var_objects[expr.name] = expr
            return self.var_indices[expr.name]

        elif isinstance(expr, Function):
            # Compile all function arguments
            arg_indices = []
            arg_consts = []
            for arg in expr.args:
                idx = self._compile_node(arg)
                arg_indices.append(idx)
                if idx == -1:
                    arg_consts.append(arg)
                else:
                    arg_consts.append(None)

            result_slot = self._alloc_slot()

            # Create a special op for functions
            op = CompiledOp(
                op="func",
                result_idx=result_slot,
                left_idx=arg_indices[0] if arg_indices else -1,
                right_idx=arg_indices[1] if len(arg_indices) > 1 else -1,
                left_const=arg_consts[0] if arg_consts else None,
                right_const=arg_consts[1] if len(arg_consts) > 1 else None,
                func_callable=expr.func,
                func_kwargs=expr.kwargs if hasattr(expr, 'kwargs') else {},
            )
            # Store extra args for functions with >2 arguments
            op._extra_arg_indices = arg_indices[2:] if len(arg_indices) > 2 else []
            op._extra_arg_consts = arg_consts[2:] if len(arg_consts) > 2 else []

            self.ops.append(op)
            self.expr_to_slot[expr_id] = result_slot
            return result_slot

        elif isinstance(expr, Expr):
            # Compile left operand
            left_idx = self._compile_node(expr.left)
            left_const = expr.left if left_idx == -1 else None

            # Compile right operand if present
            if expr.right is not None:
                right_idx = self._compile_node(expr.right)
                right_const = expr.right if right_idx == -1 else None
            else:
                right_idx = -1
                right_const = None

            result_slot = self._alloc_slot()

            # Check if this is an atom (callable Expr like norm, trace, etc.)
            atom_callable = None
            if callable(expr) and expr.op not in _BINARY_OPS and expr.op not in _UNARY_OPS:
                atom_callable = expr

            # Special handling for getitem - store the key
            getitem_key = None
            if expr.op == "getitem":
                getitem_key = expr.right
                right_idx = -1  # Key is not a compiled expression
                right_const = None

            op = CompiledOp(
                op=expr.op,
                result_idx=result_slot,
                left_idx=left_idx,
                right_idx=right_idx,
                left_const=left_const,
                right_const=right_const,
                atom_callable=atom_callable,
                getitem_key=getitem_key,
            )
            self.ops.append(op)
            self.expr_to_slot[expr_id] = result_slot
            return result_slot

        elif isinstance(expr, dict):
            # Compile dict expressions - each value becomes a compiled subexpr
            # For now, return -1 and let it be handled as a constant
            # (Full dict compilation would require more complex handling)
            return -1

        elif isinstance(expr, Iterable) and not isinstance(expr, (np.ndarray, str)):
            # Lists/tuples of expressions - compile each element
            # For now, treat as constant
            return -1

        else:
            # Constants (int, float, ndarray) - return -1 to indicate inline
            return -1


def compile_expression(expr) -> CompiledExpression:
    """
    Compile an expression tree into a fast-evaluating CompiledExpression.

    Args:
        expr: An expression tree (Expr, Variable, Function, or constant)

    Returns:
        CompiledExpression that can be called with a var_dict

    Example:
        >>> x = Variable((3,), name="x")
        >>> y = Variable((3,), name="y")
        >>> expr = x + y * 2
        >>> compiled = compile_expression(expr)
        >>> result = compiled({"x": np.array([1,2,3]), "y": np.array([4,5,6])})
    """
    compiler = ExpressionCompiler()
    return compiler.compile(expr)


def eval_compiled(compiled: CompiledExpression, var_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Evaluate a compiled expression with the given variable values.

    This is the fast path - no recursion, no isinstance checks per node,
    just sequential operation execution.
    """
    # Allocate or reuse workspace
    workspace = [None] * compiled.num_slots

    # Load variables into their slots
    for name, slot in compiled.var_indices.items():
        workspace[slot] = var_dict[name]

    # Execute operations sequentially
    for op in compiled.ops:
        # Get left operand
        if op.left_idx >= 0:
            left = workspace[op.left_idx]
        else:
            left = op.left_const

        # Get right operand
        if op.right_idx >= 0:
            right = workspace[op.right_idx]
        elif op.right_const is not None:
            right = op.right_const
        else:
            right = None

        # Execute operation
        if op.op in _BINARY_OPS:
            result = _BINARY_OPS[op.op](left, right)
        elif op.op in _UNARY_OPS:
            result = _UNARY_OPS[op.op](left)
        elif op.op == "getitem":
            result = left[op.getitem_key]
        elif op.atom_callable is not None:
            # Atom operations (norm, trace, det, etc.)
            if right is None:
                result = op.atom_callable(left)
            else:
                result = op.atom_callable(left, right)
        elif op.op == "func":
            # User-defined function
            args = [left]
            if right is not None:
                args.append(right)
            # Handle extra args for functions with >2 parameters
            if hasattr(op, '_extra_arg_indices'):
                for idx, const in zip(op._extra_arg_indices, op._extra_arg_consts):
                    if idx >= 0:
                        args.append(workspace[idx])
                    else:
                        args.append(const)
            kwargs = op.func_kwargs or {}
            result = op.func_callable(*args, **kwargs)
        else:
            raise NotImplementedError(f"Compiled execution for op: {op.op}")

        workspace[op.result_idx] = result

    return workspace[compiled.result_idx]


def eval_compiled_with_values(compiled: CompiledExpression) -> np.ndarray:
    """
    Evaluate a compiled expression using Variable._value attributes.

    This mirrors the use_value=True behavior of the original eval_expression.
    """
    # Build var_dict from Variable objects
    # Note: This requires the compiler to store Variable references
    # For now, we'll need to pass this info through
    raise NotImplementedError(
        "eval_compiled_with_values requires Variable object references. "
        "Use eval_compiled with an explicit var_dict instead."
    )


# Convenience function that matches original API
def eval_expression_compiled(expr, var_dict, use_value=False, _cache={}):
    """
    Drop-in replacement for eval_expression that uses compilation.

    Caches compiled expressions by expression id for repeated evaluations.

    Args:
        expr: Expression tree to evaluate
        var_dict: Dictionary mapping variable names to values
        use_value: If True, use Variable._value (falls back to original impl)
        _cache: Internal cache for compiled expressions

    Returns:
        Evaluated result as numpy array
    """
    if use_value:
        # Fall back to original for use_value mode
        from .parser import eval_expression
        return eval_expression(expr, var_dict, use_value=True)

    # Check cache
    expr_id = id(expr)
    if expr_id not in _cache:
        _cache[expr_id] = compile_expression(expr)

    compiled = _cache[expr_id]
    return eval_compiled(compiled, var_dict)


def clear_compilation_cache():
    """Clear the expression compilation cache."""
    eval_expression_compiled.__defaults__[0].clear()


# =============================================================================
# CODEGEN COMPILER - Generates actual Python code for maximum speed
# =============================================================================

class CodegenCompiler:
    """
    Compiles expression trees into Python source code.

    This is the fastest approach as it generates actual Python code that
    can be compiled once and executed at native Python speed, with no
    interpreter overhead per operation.
    """

    def __init__(self):
        self.code_lines: List[str] = []
        self.temp_counter = 0
        self.expr_to_temp: Dict[int, str] = {}
        self.var_names: List[str] = []
        self.constants: Dict[str, Any] = {}
        self.const_counter = 0
        self.atoms: Dict[str, Callable] = {}
        self.atom_counter = 0
        self.funcs: Dict[str, Tuple[Callable, Dict]] = {}
        self.func_counter = 0

    def _alloc_temp(self) -> str:
        """Allocate a new temporary variable name."""
        name = f"_t{self.temp_counter}"
        self.temp_counter += 1
        return name

    def _alloc_const(self, value: Any) -> str:
        """Allocate a constant and return its reference name."""
        name = f"_c{self.const_counter}"
        self.const_counter += 1
        self.constants[name] = value
        return name

    def _alloc_atom(self, atom: Callable) -> str:
        """Allocate an atom callable and return its reference name."""
        name = f"_atom{self.atom_counter}"
        self.atom_counter += 1
        self.atoms[name] = atom
        return name

    def _alloc_func(self, func: Callable, kwargs: Dict) -> str:
        """Allocate a function callable and return its reference name."""
        name = f"_func{self.func_counter}"
        self.func_counter += 1
        self.funcs[name] = (func, kwargs)
        return name

    def compile(self, expr) -> Callable:
        """
        Compile an expression tree into a callable Python function.

        Returns:
            A function that takes (var_dict) and returns the result.
        """
        self.code_lines = []
        self.temp_counter = 0
        self.expr_to_temp = {}
        self.var_names = []
        self.constants = {}
        self.const_counter = 0
        self.atoms = {}
        self.atom_counter = 0
        self.funcs = {}
        self.func_counter = 0

        result_name = self._emit_node(expr)

        # Build list of constants/atoms/funcs for direct indexing (faster than dict)
        const_list = [self.constants[f"_c{i}"] for i in range(self.const_counter)]
        atom_list = [self.atoms[f"_atom{i}"] for i in range(self.atom_counter)]
        func_list = [self.funcs[f"_func{i}"] for i in range(self.func_counter)]

        # Build variable extraction code at start of function
        var_extractions = []
        for i, name in enumerate(self.var_names):
            var_extractions.append(f"_v{i} = _vars['{name}']")

        # Build the function with list indexing instead of dict lookups
        extraction_code = "\n    ".join(var_extractions) if var_extractions else "pass"
        func_body = "\n    ".join(self.code_lines)
        func_code = f"""def _compiled_eval(_vars, _C, _A, _F):
    {extraction_code}
    {func_body}
    return {result_name}
"""

        # Compile and execute to get the function
        local_ns = {}
        exec(func_code, {"__builtins__": {}}, local_ns)
        inner_func = local_ns["_compiled_eval"]

        # Capture lists in closure (tuple is slightly faster)
        const_tuple = tuple(const_list)
        atom_tuple = tuple(atom_list)
        func_tuple = tuple(func_list)

        def compiled_eval(var_dict):
            return inner_func(var_dict, const_tuple, atom_tuple, func_tuple)

        # Store metadata for debugging
        compiled_eval._source = func_code
        compiled_eval._var_names = list(self.var_names)

        return compiled_eval

    def _emit_node(self, expr) -> str:
        """
        Emit code for a node and return the variable name holding the result.
        """
        # Check if already emitted (CSE)
        expr_id = id(expr)
        if expr_id in self.expr_to_temp:
            return self.expr_to_temp[expr_id]

        if isinstance(expr, Variable):
            # Variables come from var_dict, extracted at function start
            if expr.name not in self.var_names:
                self.var_names.append(expr.name)
            var_idx = self.var_names.index(expr.name)
            result = f"_v{var_idx}"
            self.expr_to_temp[expr_id] = result
            return result

        elif isinstance(expr, Function):
            # Emit all arguments
            arg_names = [self._emit_node(arg) for arg in expr.args]

            # Allocate function reference
            kwargs = expr.kwargs if hasattr(expr, 'kwargs') else {}
            func_idx = self.func_counter
            self._alloc_func(expr.func, kwargs)

            temp = self._alloc_temp()
            args_str = ", ".join(arg_names)

            if kwargs:
                self.code_lines.append(f"{temp} = _F[{func_idx}][0]({args_str}, **_F[{func_idx}][1])")
            else:
                self.code_lines.append(f"{temp} = _F[{func_idx}][0]({args_str})")

            self.expr_to_temp[expr_id] = temp
            return temp

        elif isinstance(expr, Expr):
            # Emit left operand
            left_name = self._emit_node(expr.left)

            # Emit right operand if present
            if expr.right is not None and expr.op != "getitem":
                right_name = self._emit_node(expr.right)
            else:
                right_name = None

            temp = self._alloc_temp()

            # Generate code based on operation
            if expr.op == "add":
                self.code_lines.append(f"{temp} = {left_name} + {right_name}")
            elif expr.op == "sub":
                self.code_lines.append(f"{temp} = {left_name} - {right_name}")
            elif expr.op == "mul":
                self.code_lines.append(f"{temp} = {left_name} * {right_name}")
            elif expr.op == "div":
                self.code_lines.append(f"{temp} = {left_name} / {right_name}")
            elif expr.op == "pow":
                self.code_lines.append(f"{temp} = {left_name} ** {right_name}")
            elif expr.op == "matmul":
                self.code_lines.append(f"{temp} = {left_name} @ {right_name}")
            elif expr.op == "neg":
                self.code_lines.append(f"{temp} = -{left_name}")
            elif expr.op == "transpose":
                self.code_lines.append(f"{temp} = {left_name}.T")
            elif expr.op == "flatten":
                self.code_lines.append(f"{temp} = {left_name}.flatten()")
            elif expr.op == "getitem":
                # Store the key as a constant (use index for tuple access)
                key_idx = self.const_counter
                self._alloc_const(expr.right)
                self.code_lines.append(f"{temp} = {left_name}[_C[{key_idx}]]")
            elif callable(expr):
                # Atom operation (norm, trace, etc.)
                atom_idx = self.atom_counter
                self._alloc_atom(expr)
                if right_name is None:
                    self.code_lines.append(f"{temp} = _A[{atom_idx}]({left_name})")
                else:
                    self.code_lines.append(f"{temp} = _A[{atom_idx}]({left_name}, {right_name})")
            else:
                raise NotImplementedError(f"Codegen for op: {expr.op}")

            self.expr_to_temp[expr_id] = temp
            return temp

        elif isinstance(expr, dict):
            # Compile dict - emit each value
            result_parts = []
            for k, v in expr.items():
                v_name = self._emit_node(v)
                key_idx = self.const_counter
                self._alloc_const(k)
                result_parts.append(f"_C[{key_idx}]: {v_name}")

            temp = self._alloc_temp()
            self.code_lines.append(f"{temp} = {{{', '.join(result_parts)}}}")
            self.expr_to_temp[expr_id] = temp
            return temp

        elif isinstance(expr, (list, tuple)) and not isinstance(expr, np.ndarray):
            # Compile list/tuple
            elem_names = [self._emit_node(e) for e in expr]
            temp = self._alloc_temp()

            if isinstance(expr, tuple):
                self.code_lines.append(f"{temp} = ({', '.join(elem_names)},)")
            else:
                self.code_lines.append(f"{temp} = [{', '.join(elem_names)}]")

            self.expr_to_temp[expr_id] = temp
            return temp

        else:
            # Constant - store in constants tuple (use index for access)
            const_idx = self.const_counter
            self._alloc_const(expr)
            return f"_C[{const_idx}]"


def compile_to_function(expr) -> Callable:
    """
    Compile an expression tree into a Python function.

    This is the most optimized compilation method - it generates
    actual Python code that runs at native speed.

    Args:
        expr: An expression tree (Expr, Variable, Function, or constant)

    Returns:
        A callable that takes a var_dict and returns the result

    Example:
        >>> x = Variable((3,), name="x")
        >>> y = Variable((3,), name="y")
        >>> expr = x + y * 2
        >>> f = compile_to_function(expr)
        >>> result = f({"x": np.array([1,2,3]), "y": np.array([4,5,6])})
    """
    compiler = CodegenCompiler()
    return compiler.compile(expr)


# Cache for codegen-compiled functions
_codegen_cache: Dict[int, Callable] = {}


def eval_expression_codegen(expr, var_dict, use_value=False):
    """
    Evaluate an expression using codegen compilation.

    This is the fastest evaluation method, caching compiled functions.

    Args:
        expr: Expression tree to evaluate
        var_dict: Dictionary mapping variable names to values
        use_value: If True, use Variable._value (falls back to original impl)

    Returns:
        Evaluated result as numpy array
    """
    if use_value:
        from .parser import eval_expression
        return eval_expression(expr, var_dict, use_value=True)

    expr_id = id(expr)
    if expr_id not in _codegen_cache:
        _codegen_cache[expr_id] = compile_to_function(expr)

    return _codegen_cache[expr_id](var_dict)


def clear_codegen_cache():
    """Clear the codegen compilation cache."""
    global _codegen_cache
    _codegen_cache = {}
