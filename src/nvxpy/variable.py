import threading
import logging

import autograd.numpy as np

from .expression import BaseExpr
from .constraint import Constraint
from .constants import Curvature as C

# Thread-local storage for variable ID counter
_thread_local = threading.local()
logger = logging.getLogger(__name__)


def _get_next_id() -> int:
    """Get the next variable ID in a thread-safe manner."""
    if not hasattr(_thread_local, "var_id"):
        _thread_local.var_id = 0
    current = _thread_local.var_id
    _thread_local.var_id += 1
    return current


def reset_variable_ids() -> None:
    """Reset the variable ID counter for the current thread."""
    _thread_local.var_id = 0


class Variable(BaseExpr):
    __array_priority__ = 100

    def __init__(
        self,
        shape: tuple[int, ...] = (1,),
        name: str | None = None,
        symmetric: bool = False,
        PSD: bool = False,
        NSD: bool = False,
        pos: bool = False,
        neg: bool = False,
        binary: bool = False,
        integer: bool = False,
    ) -> None:
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple")
        if len(shape) == 0:
            raise ValueError("Shape must be non-empty")
        if not all(isinstance(s, int) for s in shape):
            raise TypeError("Shape must be a tuple of integers")
        if not all(s > 0 for s in shape):
            raise ValueError("Shape must be a tuple of positive integers")
        if len(shape) not in (1, 2):
            raise ValueError("Shape must be a tuple of length 1 or 2")

        var_id = _get_next_id()
        self.name = name if name else f"x{var_id}"
        self.shape = shape
        self.size = int(np.prod(shape))
        self._value = None
        self._id = var_id

        self.constraints = []

        if symmetric:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("Symmetric variable must be a square matrix")
            U_inds = np.triu_indices(shape[0], k=1)
            L_inds = np.tril_indices(shape[0], k=-1)
            self.constraints.append(Constraint(self[U_inds], "==", self[L_inds]))

        if PSD:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("PSD variable must be a square matrix")
            if neg:
                raise ValueError("PSD variable cannot be negative")
            self.constraints.append(Constraint(self, ">>", 0))

        if NSD:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("NSD variable must be a square matrix")
            if pos:
                raise ValueError("NSD variable cannot be positive")
            self.constraints.append(Constraint(self, "<<", 0))

        if pos:
            if neg or NSD:
                raise ValueError("Positive variable cannot be NSD or negative")
            self.constraints.append(Constraint(self, ">=", 0))

        if neg:
            if PSD or pos:
                raise ValueError("Negative variable cannot be PSD or positive")
            self.constraints.append(Constraint(self, "<=", 0))

        if binary:
            if neg or PSD or NSD:
                raise ValueError("Binary variable cannot be negative, PSD, or NSD")
            if pos:
                logger.warning(
                    "Setting binary variable to be positive is redundant, can only be 0 or 1"
                )
            if integer:
                logger.warning("Setting binary variable to be integer is redundant")
            self.constraints.append(Constraint(self, ">=", 0))
            self.constraints.append(Constraint(self, "<=", 1))

        self.is_integer = bool(integer) or bool(binary)
        self.is_binary = bool(binary)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        arr = np.array(val)
        if arr.size != self.size:
            raise ValueError(
                f"Cannot assign value with {arr.size} elements to variable with shape {self.shape} ({self.size} elements)"
            )
        self._value = arr.reshape(self.shape)

    def __repr__(self) -> str:
        return f"Var({self.name}, shape={self.shape})"

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    def curvature(self):
        return C.AFFINE
