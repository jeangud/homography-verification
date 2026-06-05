"""Homogeneous coordinates conversion functions."""

import numpy as np


def to_homogeneous(xy):
    """Converts from (x,y) coordinates to homogeneous (x,y,1) coordinates."""
    return np.vstack((xy, np.ones(xy.shape[1])))


def from_homogeneous(xyz):
    """Converts from homogeneous (x,y,1) coordinates to (x,y) coordinates."""
    return xyz[:2] / xyz[2]
