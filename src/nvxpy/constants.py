from enum import StrEnum, Enum, auto

# Small epsilon to prevent numerical issues (e.g., division by zero, gradient singularities)
EPSILON = 1e-12

# Default tolerances for optimization
DEFAULT_PROJECTION_TOL = 1e-6  # Tolerance for projection constraints
DEFAULT_DISCRETE_TOL = 1e-6   # Tolerance for discrete set membership
DEFAULT_SOLVER_TOL = 1e-8     # Default solver function tolerance
DEFAULT_INT_TOL = 1e-5        # Integer feasibility tolerance for BNB


class Solver(StrEnum):
    # Gradient-free methods
    NELDER_MEAD = "Nelder-Mead"
    POWELL = "Powell"
    COBYLA = "COBYLA"
    COBYQA = "COBYQA"

    # Gradient-based methods
    CG = "CG"
    BFGS = "BFGS"
    LBFGSB = "L-BFGS-B"
    TNC = "TNC"
    SLSQP = "SLSQP"

    # Hessian-based methods
    NEWTON_CG = "Newton-CG"
    DOGLEG = "dogleg"
    TRUST_NCG = "trust-ncg"
    TRUST_KRYLOV = "trust-krylov"
    TRUST_EXACT = "trust-exact"
    TRUST_CONSTR = "trust-constr"

    # Global optimizers
    DIFF_EVOLUTION = "differential_evolution"
    DUAL_ANNEALING = "dual_annealing"
    SHGO = "shgo"
    BASINHOPPING = "basinhopping"

    # Other
    IPOPT = "IPOPT"
    BNB = "BnB"  # Branch-and-Bound MINLP solver


class Curvature(Enum):
    CONSTANT = auto()
    AFFINE = auto()
    CONVEX = auto()
    CONCAVE = auto()
    UNKNOWN = auto()
    
