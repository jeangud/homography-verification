"""Utilities to manipulate image coordinates."""


def to_xy(i: float, j: float):
    """Converts from pixel indices (i,j) to spatial (x,y) coordinates.

    This transform accounts for pixel center location.

    See:
     1. https://boofcv.org/index.php?title=Coordinate_Systems
     2. https://ch.mathworks.com/help/images/image-coordinate-systems.html
    """
    return j + 0.5, i + 0.5  # (i, j) represents pixel center


def to_ij(x: float, y: float):
    """Converts from spatial (x,y) coordinates to pixel indices (i,j).

    This transform accounts for pixel center location.

    See:
     1. https://boofcv.org/index.php?title=Coordinate_Systems
     2. https://ch.mathworks.com/help/images/image-coordinate-systems.html
    """
    return y - 0.5, x - 0.5  # (x, y) represents pixel top-left corner
