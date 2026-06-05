from unittest.mock import Mock

from vnn.transforms.transform import Transform
from vnn.transforms.transform_with_bounds import TransformWithBounds


def test_transform_with_bounds_str():
    # Mocking a transform since we just want to test the string representation
    mock_transform = Mock(spec=Transform)
    mock_transform.__str__ = Mock(return_value="MockTransform")

    twb = TransformWithBounds(
        transform=mock_transform, lower_bound=-1.5, upper_bound=2.0
    )

    assert str(twb) == "MockTransform[-1.50,2.00]"
