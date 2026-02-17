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
    A set of allowed discrete values or n-dimensional points.

    Used to constrain a variable to take values from a specific discrete set.
    Supports both scalar values and n-dimensional points.

    Example:
        # Scalar variable with discrete values
        x = Variable(integer=True)
        constraint = x ^ DiscreteSet([1, 10, 100, 9, -2])
        # or using the list shorthand:
        constraint = x ^ [1, 10, 100, 9, -2]

        # Also works with floats:
        y = Variable()
        constraint = y ^ [0.1, 0.5, 1.0, 2.5]

        # n-dimensional points for vector variables
        z = Variable(shape=(2,))
        constraint = z ^ [[1, 3], [3, 6], [6, 2]]  # z can be one of these 2D points
    """

    def __init__(
        self,
        values: Sequence[Union[int, float, Sequence]],
        tolerance: float = DEFAULT_DISCRETE_TOL,
    ):
        """
        Create a DiscreteSet with allowed values or points.

        Args:
            values: A sequence of allowed values (scalars or n-D points)
            tolerance: Tolerance for checking membership (default: DEFAULT_DISCRETE_TOL)
        """
        if not values:
            raise ValueError("DiscreteSet must contain at least one value")

        # Check if we have scalars or n-D points
        first_item = values[0]
        if isinstance(first_item, (int, float)):
            # Scalar mode
            self._point_dim = 1
            # Keep original values, sort them, remove duplicates within tolerance
            sorted_vals = sorted(set(float(v) for v in values))
            # Remove duplicates that are too close
            unique_vals = []
            for v in sorted_vals:
                if not unique_vals or not np.isclose(
                    v, unique_vals[-1], atol=tolerance
                ):
                    unique_vals.append(v)
            self._values = tuple(unique_vals)
        else:
            # n-D points mode
            parsed_points = []
            first_shape = None
            for p in values:
                arr = np.array(p, dtype=float)
                if first_shape is None:
                    first_shape = arr.shape
                elif arr.shape != first_shape:
                    raise ValueError(
                        f"All points must have the same shape. Got {first_shape} and {arr.shape}"
                    )
                parsed_points.append(tuple(arr.flatten()))

            self._point_dim = len(parsed_points[0]) if parsed_points else 0

            # Remove duplicates (within tolerance)
            unique_points = []
            for p in parsed_points:
                is_duplicate = False
                for existing in unique_points:
                    if all(
                        np.isclose(p[i], existing[i], atol=tolerance)
                        for i in range(len(p))
                    ):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(p)
            self._values = tuple(unique_points)

        self._tolerance = tolerance
        if not self._values:
            raise ValueError("DiscreteSet must contain at least one value")

        # Build name based on mode
        if self._point_dim == 1:
            super().__init__(name=f"DiscreteSet({list(self._values)})")
        else:
            super().__init__(name=f"DiscreteSet({[list(p) for p in self._values]})")

    @property
    def values(self) -> tuple:
        """The tuple of allowed values (scalars) or points (n-D tuples)."""
        return self._values

    @property
    def point_dim(self) -> int:
        """The dimensionality of each point (1 for scalars)."""
        return self._point_dim

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
            ValueError: If variable size doesn't match point dimensionality
        """
        # Check dimension compatibility
        var_size = getattr(var, "size", 1)
        if var_size != self._point_dim:
            raise ValueError(
                f"Variable size ({var_size}) does not match DiscreteSet point dimension ({self._point_dim}). "
                f"For a variable of size {var_size}, provide points of dimension {var_size}."
            )

        # Check if variable is integer-constrained (only for scalar mode)
        if self._point_dim == 1 and getattr(var, "is_integer", False):
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
        if self._point_dim == 1:
            return any(
                np.isclose(float(value), v, atol=self._tolerance) for v in self._values
            )
        else:
            arr = np.array(value).flatten()
            if len(arr) != self._point_dim:
                return False
            return any(
                all(
                    np.isclose(arr[i], p[i], atol=self._tolerance)
                    for i in range(self._point_dim)
                )
                for p in self._values
            )

    def nearest(self, value):
        """Find the nearest value/point in the set."""
        if self._point_dim == 1:
            return min(self._values, key=lambda v: abs(float(value) - v))
        else:
            arr = np.array(value).flatten()
            return min(
                self._values,
                key=lambda p: sum((arr[i] - p[i]) ** 2 for i in range(self._point_dim)),
            )

    def values_below(self, value: float) -> tuple:
        """Return all values in the set strictly below the given value (scalar mode only)."""
        if self._point_dim != 1:
            raise ValueError("values_below is only supported for scalar DiscreteSet")
        return tuple(v for v in self._values if v < value - self._tolerance)

    def values_above(self, value: float) -> tuple:
        """Return all values in the set strictly above the given value (scalar mode only)."""
        if self._point_dim != 1:
            raise ValueError("values_above is only supported for scalar DiscreteSet")
        return tuple(v for v in self._values if v > value + self._tolerance)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __repr__(self):
        if self._point_dim == 1:
            return f"DiscreteSet({list(self._values)})"
        else:
            return f"DiscreteSet({[list(p) for p in self._values]})"


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
        self, ranges: Sequence[Sequence], tolerance: float = DEFAULT_DISCRETE_TOL
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
                merged_ranges[-1] = Range(
                    merged_ranges[-1].lb, max(merged_ranges[-1].ub, r.ub)
                )
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


def _coerce_to_discrete_set(
    obj, var_size: int = 1
) -> Union[DiscreteSet, DiscreteRanges]:
    """
    Coerce a list/tuple to DiscreteSet or DiscreteRanges based on variable size.

    The interpretation depends on matching variable size to element shape:

    For scalar variables (var_size=1):
        x ^ [1, 2, 3]              -> DiscreteSet (all scalars)
        x ^ [[0, 5], [10, 15]]     -> DiscreteRanges (each is a [lb, ub] range)

    For n-dimensional variables (var_size=n):
        x ^ [[1,2], [3,4], [5,6]]  -> DiscreteSet with n-D points if each element has n values
        x ^ [[0,5], [10,15]]       -> DiscreteRanges if var_size=1 (ranges)
                                      DiscreteSet with 2D points if var_size=2

    Args:
        obj: The list/tuple to coerce
        var_size: The size of the variable being constrained (default: 1 for scalar)

    Returns:
        DiscreteSet or DiscreteRanges
    """
    if isinstance(obj, (DiscreteSet, DiscreteRanges)):
        return obj
    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError("Cannot create discrete set from empty sequence")

        # Check if all items are scalars -> DiscreteSet (scalar mode)
        all_scalars = all(isinstance(item, (int, float)) for item in obj)
        if all_scalars:
            return DiscreteSet(obj)

        # Check if all items are sequences
        all_sequences = all(isinstance(item, (list, tuple)) for item in obj)
        if not all_sequences:
            raise ValueError(
                "Cannot mix discrete values and sequences. "
                "Use either all scalars [1, 2, 3] or all sequences [[1,2], [3,4]]."
            )

        # All items are sequences - determine if ranges or points based on var_size
        # Get the length of the first element
        first_len = len(obj[0])

        # Check all elements have consistent length
        if not all(len(item) == first_len for item in obj):
            raise ValueError(
                "All elements must have the same length. "
                f"Got lengths: {[len(item) for item in obj]}"
            )

        # Decision logic:
        # - If var_size == 1 and element length == 2 -> DiscreteRanges (scalar with [lb, ub] ranges)
        # - If var_size == element length -> DiscreteSet with n-D points
        # - Otherwise -> error

        if var_size == 1 and first_len == 2:
            # Scalar variable with 2-element lists -> interpret as ranges
            return DiscreteRanges(obj)
        elif var_size == first_len:
            # n-D variable with n-element lists -> interpret as discrete n-D points
            return DiscreteSet(obj)
        else:
            raise ValueError(
                f"Element length ({first_len}) does not match variable size ({var_size}). "
                f"For a variable of size {var_size}, provide elements of length {var_size} for discrete points, "
                f"or use a scalar variable (size 1) with 2-element ranges [[lb, ub], ...]."
            )
    raise TypeError(f"Cannot convert {type(obj)} to DiscreteSet or DiscreteRanges")
