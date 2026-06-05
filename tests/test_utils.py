import argparse
import logging
from enum import Enum, auto
from pathlib import Path
from unittest.mock import patch

import pytest

from vnn import utils


class DummyEnum(Enum):
    A = auto()
    B = auto()


def test_enum_action_valid_string():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=DummyEnum, action=utils.EnumAction)
    args = parser.parse_args(["--test", "A"])
    assert args.test == DummyEnum.A


def test_enum_action_valid_list():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", nargs="+", type=DummyEnum, action=utils.EnumAction)
    args = parser.parse_args(["--test", "A", "B"])
    assert args.test == [DummyEnum.A, DummyEnum.B]


def test_enum_action_invalid_type():
    with pytest.raises(ValueError, match="type must be assigned an Enum"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", action=utils.EnumAction)

    with pytest.raises(TypeError, match="type must be an Enum"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", type=int, action=utils.EnumAction)


def test_enum_action_invalid_value():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=DummyEnum, action=utils.EnumAction)

    with pytest.raises(SystemExit):  # argparse exits on invalid choice
        parser.parse_args(["--test", "C"])


def test_enum_action_call_none():
    parser = argparse.ArgumentParser()
    action = utils.EnumAction(option_strings=["--test"], dest="test", type=DummyEnum)
    namespace = argparse.Namespace()

    with pytest.raises(argparse.ArgumentTypeError, match="You need to pass a value"):
        action(parser, namespace, None, option_string="--test")


def test_enum_action_call_unsupported_type():
    parser = argparse.ArgumentParser()
    action = utils.EnumAction(option_strings=["--test"], dest="test", type=DummyEnum)
    namespace = argparse.Namespace()

    with pytest.raises(
        argparse.ArgumentTypeError, match="EnumAction value type not supported"
    ):
        action(parser, namespace, 123, option_string="--test")


def test_get_logging_levels():
    levels = utils.get_logging_levels()
    assert isinstance(levels, list)
    assert "DEBUG" in levels
    assert "INFO" in levels


def test_setup_logging():
    # Basic check that setup_logging executes without error
    utils.setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    # With file path
    with patch("logging.FileHandler") as m_file_handler:
        utils.setup_logging("INFO", path_log=Path("dummy.log"))
        m_file_handler.assert_called_once()

    # Clean up mock handler left on root logger by the patched FileHandler
    logging.getLogger().handlers.clear()
