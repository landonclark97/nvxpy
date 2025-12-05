class Set:
    """Base class for set constraints.

    Sets are used with the ^ operator to constrain variables to specific domains.
    Subclasses include DiscreteSet (finite set of values), DiscreteRanges
    (union of intervals), and geometric sets like SO3.

    Example:
        x = Variable()
        constraint = x ^ [1, 2, 3]  # x must be in {1, 2, 3}
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Set({self.name})"

    def constrain(self, var):
        """Create a constraint that var belongs to this set."""
        raise NotImplementedError("Subclasses must implement this method")
