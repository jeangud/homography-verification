"""The different types of geometric transformations."""

from enum import Enum, auto


class TransformType(Enum):
    """Enumeration of the different types of geometric transformations."""

    # Affine
    ROTATE = auto()
    SCALE = auto()
    SHEAR_X = auto()
    TRANSLATE_X = auto()
    TRANSLATE_Y = auto()
    # Non-affine
    H_ROLL = auto()
    H_PITCH = auto()
    H_YAW = auto()
    H_X = auto()
    H_Y = auto()
    H_Z = auto()

    def __str__(self):
        return self.name
