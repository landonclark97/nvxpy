from typing import TYPE_CHECKING

from .constants import Curvature as C

if TYPE_CHECKING:
    from .expression import ExprLike
    from .set import Set


class Constraint:
    """
    A constraint in an optimization problem.

    Operators:
        >= : Greater than or equal constraint
        <= : Less than or equal constraint
        == : Equality constraint
        >> : Positive semidefinite (PSD) constraint
        << : Negative semidefinite (NSD) constraint
        <- : Projection constraint (variable is projected to a set)
        in : Discrete set membership constraint
    """

    def __init__(self, left: "ExprLike", op: str, right: "ExprLike | Set") -> None:
        valid_ops = [">=", "<=", "==", ">>", "<<", "<-", "in"]
        if op not in valid_ops:
            raise ValueError(f"Invalid constraint operator '{op}'. Must be one of: {valid_ops}")
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"Constraint({self.left} {self.op} {self.right})"

    @property
    def curvature(self):
        # Discrete set membership constraints are non-convex
        if self.op == "in":
            return C.UNKNOWN

        res = self.right - self.left if self.op in [">=", "==", ">>", "<-"] else self.left - self.right
        curvature = res.curvature
        return curvature if curvature in (C.CONSTANT, C.AFFINE, C.CONVEX) else C.UNKNOWN
