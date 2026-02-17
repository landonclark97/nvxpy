from enum import StrEnum, Enum, auto

# Small epsilon to prevent numerical issues (e.g., division by zero, gradient singularities)
EPSILON = 1e-12

# Default tolerances for optimization
DEFAULT_PROJECTION_TOL = 5e-3  # Tolerance for projection constraints in main solve
DEFAULT_DISCRETE_TOL = 1e-6  # Tolerance for discrete set membership
DEFAULT_SOLVER_TOL = 1e-8  # Default solver function tolerance
DEFAULT_INT_TOL = 1e-5  # Integer feasibility tolerance for BNB

# Branch-and-Bound tolerances
DEFAULT_ABS_GAP = 1e-6  # Absolute optimality gap tolerance
DEFAULT_REL_GAP = 1e-4  # Relative optimality gap tolerance
DEFAULT_NLP_FTOL = 1e-9  # NLP function tolerance (high precision)
DEFAULT_BRANCH_FTOL = 1e-6  # NLP tolerance for branching (quick/loose)
DEFAULT_CON_TOL = 1e-4  # Constraint violation tolerance (active constraint detection)
DEFAULT_NEAR_ZERO = 1e-10  # Near-zero comparison tolerance
DEFAULT_FEASIBILITY_TOL = 1e-6  # Constraint feasibility tolerance

# Numerical differentiation
DEFAULT_GRADIENT_EPS = 1e-8  # Epsilon for numerical gradient approximation


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
    KNITRO = "KNITRO"  # Commercial NLP/MINLP solver
    BNB = "BnB"  # Branch-and-Bound MINLP solver


class Curvature(Enum):
    CONSTANT = auto()
    AFFINE = auto()
    CONVEX = auto()
    CONCAVE = auto()
    UNKNOWN = auto()
