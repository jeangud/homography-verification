from vnn.transforms import image


def test_to_xy():
    # Test standard translation (origin 0,0 maps to center 0.5, 0.5)
    # y = j + 0.5, x = i + 0.5
    y, x = image.to_xy(0, 0)
    assert x == 0.5
    assert y == 0.5

    y, x = image.to_xy(2, 3)
    assert x == 2.5
    assert y == 3.5


def test_to_ij():
    # Test reverse mapping
    # i = y - 0.5, j = x - 0.5
    i, j = image.to_ij(1.5, 2.5)
    assert i == 2.0
    assert j == 1.0

    i, j = image.to_ij(0.5, 0.5)
    assert i == 0.0
    assert j == 0.0
