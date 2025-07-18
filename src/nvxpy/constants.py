from enum import StrEnum, Enum, auto


class Solver(StrEnum):
    SLSQP = "SLSQP"
    IPOPT = "IPOPT"
    COBYLA = "COBYLA"
    NELDER_MEAD = "NELDER_MEAD"
    BFGS = "BFGS"
    LBFGSB = "L-BFGS-B"
    TNC = "TNC"
    TRUST_CONSTR = "trust-constr"


class Curvature(Enum):
    CONSTANT = auto()
    AFFINE = auto()
    CONVEX = auto()
    CONCAVE = auto()
    UNKNOWN = auto()
    