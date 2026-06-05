from vnn.transforms.transform_type import TransformType


def test_transform_type_str():
    for member in TransformType:
        assert str(member) == member.name
