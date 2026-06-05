"""A transformation with its corresponding bounds."""

from dataclasses import dataclass

from .transform import Transform


@dataclass
class TransformWithBounds:
    """A transformation with its corresponding bounds."""

    transform: Transform
    lower_bound: float
    upper_bound: float

    def __str__(self):
        return f"{self.transform}[{self.lower_bound:.2f},{self.upper_bound:.2f}]"
