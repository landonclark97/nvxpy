from __future__ import annotations

from typing import Sequence, Union


from ..set import Set
from ..constraint import Constraint


class DiscreteSet(Set):
    """
    A set of allowed discrete values (integers or floats).

    Used to constrain a variable to take values from a specific discrete set.

    Example:
        x = Variable(integer=True)
        constraint = x ^ DiscreteSet([1, 10, 100, 9, -2])
        # or using the list shorthand:
        constraint = x ^ [1, 10, 100, 9, -2]

        # Also works with floats:
        y = Variable()
        constraint = y ^ [0.1, 0.5, 1.0, 2.5]
    """

    def __init__(self, values: Sequence[Union[int, float]], tolerance: float = 1e-6):
        """
        Create a DiscreteSet with allowed values.

        Args:
            values: A sequence of allowed values (int or float)
            tolerance: Tolerance for checking membership (default: 1e-6)
        """
        # Keep original values, sort them, remove duplicates within tolerance
        sorted_vals = sorted(set(float(v) for v in values))
        # Remove duplicates that are too close
        unique_vals = []
        for v in sorted_vals:
            if not unique_vals or abs(v - unique_vals[-1]) > tolerance:
                unique_vals.append(v)

        self._values = tuple(unique_vals)
        self._tolerance = tolerance
        if not self._values:
            raise ValueError("DiscreteSet must contain at least one value")
        super().__init__(name=f"DiscreteSet({list(self._values)})")

    @property
    def values(self) -> tuple:
        """The sorted tuple of allowed values."""
        return self._values

    @property
    def tolerance(self) -> float:
        """The tolerance for membership checking."""
        return self._tolerance

    def constrain(self, var):
        """
        Create a discrete membership constraint.

        Args:
            var: The variable to constrain

        Returns:
            Constraint object with "in" operator
        """
        return Constraint(var, "in", self)

    def __contains__(self, value) -> bool:
        """Check if a value is in the set (within tolerance)."""
        return any(abs(float(value) - v) <= self._tolerance for v in self._values)

    def nearest(self, value: float) -> float:
        """Find the nearest value in the set."""
        return min(self._values, key=lambda v: abs(v - value))

    def values_below(self, value: float) -> tuple:
        """Return all values in the set strictly below the given value."""
        return tuple(v for v in self._values if v < value - self._tolerance)

    def values_above(self, value: float) -> tuple:
        """Return all values in the set strictly above the given value."""
        return tuple(v for v in self._values if v > value + self._tolerance)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __repr__(self):
        return f"DiscreteSet({list(self._values)})"


# Alias for backward compatibility
IntegerSet = DiscreteSet


def _coerce_to_discrete_set(obj) -> DiscreteSet:
    """
    Coerce a list/tuple to DiscreteSet if needed.

    This helper allows the shorthand: x ^ [1, 2, 3]
    """
    if isinstance(obj, DiscreteSet):
        return obj
    if isinstance(obj, (list, tuple)):
        return DiscreteSet(obj)
    raise TypeError(f"Cannot convert {type(obj)} to DiscreteSet")
