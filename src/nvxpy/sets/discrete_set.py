from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union, Tuple, List

import autograd.numpy as np

from ..set import Set
from ..constraint import Constraint
from ..constants import DEFAULT_DISCRETE_TOL


@dataclass(frozen=True)
class Range:
    """A continuous range [lb, ub] within a MixedDiscreteSet."""
    lb: float
    ub: float

    def __post_init__(self):
        if self.lb > self.ub:
            raise ValueError(f"Range lower bound {self.lb} > upper bound {self.ub}")

    def __contains__(self, value: float) -> bool:
        return self.lb <= value <= self.ub

    def __repr__(self):
        return f"[{self.lb}, {self.ub}]"


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

    def __init__(self, values: Sequence[Union[int, float]], tolerance: float = DEFAULT_DISCRETE_TOL):
        """
        Create a DiscreteSet with allowed values.

        Args:
            values: A sequence of allowed values (int or float)
            tolerance: Tolerance for checking membership (default: DEFAULT_DISCRETE_TOL)
        """
        # Keep original values, sort them, remove duplicates within tolerance
        sorted_vals = sorted(set(float(v) for v in values))
        # Remove duplicates that are too close
        unique_vals = []
        for v in sorted_vals:
            if not unique_vals or not np.isclose(v, unique_vals[-1], atol=tolerance):
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

        Raises:
            ValueError: If var is an integer variable but set contains non-integers
        """
        # Check if variable is integer-constrained
        if getattr(var, "is_integer", False):
            non_integers = [v for v in self._values if v != int(v)]
            if non_integers:
                raise ValueError(
                    f"Cannot constrain integer variable to DiscreteSet containing "
                    f"non-integer values: {non_integers}. Either declare the variable "
                    f"as continuous or use integer values in the set."
                )
        return Constraint(var, "in", self)

    def __contains__(self, value) -> bool:
        """Check if a value is in the set (within tolerance)."""
        return any(np.isclose(float(value), v, atol=self._tolerance) for v in self._values)

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


class DiscreteRanges(Set):
    """
    A set of disjoint continuous ranges (intervals).

    Constrains a variable to be within one of several continuous ranges.
    The variable is treated as continuous once committed to a specific range.
    Useful for modeling disjunctive constraints without Big-M formulations.

    Example:
        # Time slots: 9am-12pm or 1pm-5pm
        slots = [[9, 12], [13, 17]]
        constraint = start_time ^ slots

        # Temperature zones
        zones = [[0, 10], [25, 35], [50, 60]]
        constraint = temp ^ zones
    """

    def __init__(
        self,
        ranges: Sequence[Sequence],
        tolerance: float = DEFAULT_DISCRETE_TOL
    ):
        """
        Create a DiscreteRanges with continuous ranges.

        Args:
            ranges: A sequence of 2-element [lb, ub] ranges
            tolerance: Tolerance for merging adjacent ranges
        """
        parsed_ranges: List[Range] = []

        for item in ranges:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                lb, ub = float(item[0]), float(item[1])
                parsed_ranges.append(Range(lb, ub))
            else:
                raise ValueError(
                    f"Invalid range {item}: must be a 2-element [lb, ub] sequence"
                )

        if not parsed_ranges:
            raise ValueError("DiscreteRanges must contain at least one range")

        # Sort ranges by lower bound
        parsed_ranges = sorted(parsed_ranges, key=lambda r: r.lb)

        # Merge overlapping ranges
        merged_ranges: List[Range] = []
        for r in parsed_ranges:
            if merged_ranges and r.lb <= merged_ranges[-1].ub + tolerance:
                # Overlapping or adjacent, merge
                merged_ranges[-1] = Range(merged_ranges[-1].lb, max(merged_ranges[-1].ub, r.ub))
            else:
                merged_ranges.append(r)

        self._ranges = tuple(merged_ranges)
        self._tolerance = tolerance

        # Build name
        parts = [repr(r) for r in self._ranges]
        super().__init__(name=f"DiscreteRanges({parts})")

    @property
    def ranges(self) -> Tuple[Range, ...]:
        """The sorted tuple of continuous ranges."""
        return self._ranges

    @property
    def tolerance(self) -> float:
        """The tolerance for membership checking."""
        return self._tolerance

    @property
    def num_branches(self) -> int:
        """Number of disjuncts (ranges)."""
        return len(self._ranges)

    def constrain(self, var):
        """Create a discrete ranges membership constraint."""
        return Constraint(var, "in", self)

    def __contains__(self, value: float) -> bool:
        """Check if a value is in any range."""
        return any(value in r for r in self._ranges)

    def nearest(self, value: float) -> float:
        """Find the nearest value in any range."""
        candidates = []

        for r in self._ranges:
            if value in r:
                candidates.append((0.0, value))
            else:
                dist_lb = abs(value - r.lb)
                dist_ub = abs(value - r.ub)
                if dist_lb < dist_ub:
                    candidates.append((dist_lb, r.lb))
                else:
                    candidates.append((dist_ub, r.ub))

        return min(candidates, key=lambda x: x[0])[1]

    def ranges_below(self, value: float) -> Tuple[Range, ...]:
        """Return ranges strictly below the given value."""
        ranges_below = []
        for r in self._ranges:
            if r.ub < value - self._tolerance:
                ranges_below.append(r)
            elif r.lb < value - self._tolerance:
                # Range straddles value, truncate it
                ranges_below.append(Range(r.lb, value - self._tolerance))
        return tuple(ranges_below)

    def ranges_above(self, value: float) -> Tuple[Range, ...]:
        """Return ranges strictly above the given value."""
        ranges_above = []
        for r in self._ranges:
            if r.lb > value + self._tolerance:
                ranges_above.append(r)
            elif r.ub > value + self._tolerance:
                # Range straddles value, truncate it
                ranges_above.append(Range(value + self._tolerance, r.ub))
        return tuple(ranges_above)

    def bounds(self) -> Tuple[float, float]:
        """Get the overall bounds of the set."""
        return self._ranges[0].lb, self._ranges[-1].ub

    def __repr__(self):
        return f"DiscreteRanges({[[r.lb, r.ub] for r in self._ranges]})"


def _coerce_to_discrete_set(obj) -> Union[DiscreteSet, DiscreteRanges]:
    """
    Coerce a list/tuple to DiscreteSet or DiscreteRanges if needed.

    This helper allows the shorthand:
        x ^ [1, 2, 3]              -> DiscreteSet (all scalars)
        x ^ [[0, 5], [10, 15]]     -> DiscreteRanges (all ranges)

    Note: Mixed discrete values and ranges are NOT supported.
    Use either all scalars (DiscreteSet) or all ranges (DiscreteRanges).
    """
    if isinstance(obj, (DiscreteSet, DiscreteRanges)):
        return obj
    if isinstance(obj, (list, tuple)):
        # Check if items are ranges or scalars
        has_range = any(
            isinstance(item, (list, tuple)) and len(item) == 2
            for item in obj
        )
        has_scalar = any(
            isinstance(item, (int, float))
            for item in obj
        )

        if has_range and has_scalar:
            raise ValueError(
                "Cannot mix discrete values and ranges. "
                "Use DiscreteSet for discrete values [1, 2, 3] or "
                "DiscreteRanges for ranges [[0, 5], [10, 15]]."
            )

        if has_range:
            return DiscreteRanges(obj)
        return DiscreteSet(obj)
    raise TypeError(f"Cannot convert {type(obj)} to DiscreteSet or DiscreteRanges")
