"""Tests that README example commands are valid CLI invocations.

Extracts shell commands from README.md code blocks and verifies they are
accepted by the corresponding script's argument parser.  If the README is
edited with new flags or examples, these tests will catch inconsistencies.
"""

import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _extract_commands(script_name: str) -> list[str]:
    """Extract commands from README fenced code blocks that invoke *script_name*."""
    text = README.read_text()
    code_blocks = re.findall(r"```(?:shell|python)\n(.*?)```", text, re.DOTALL)
    commands = []
    for block in code_blocks:
        for line in block.strip().splitlines():
            line = line.strip()
            if script_name not in line:
                continue
            # Only keep lines that look like actual commands (skip log output)
            if line.startswith(("python ", "python3 ")) or line.startswith("scripts/"):
                commands.append(line)
    return commands


def _command_to_argv(command: str) -> list[str]:
    """Convert a shell command string to an argv list for ``parse_arguments``.

    Strips the interpreter and script path, returning only the flags/arguments
    (matching the ``parse_arguments(argv=None)`` convention where ``None``
    causes argparse to read ``sys.argv[1:]``).
    """
    tokens = shlex.split(command)
    # Strip leading 'python' if present
    if tokens and tokens[0] == "python":
        tokens = tokens[1:]
    # Drop the script path — parse_arguments expects only flags
    tokens = tokens[1:]
    return tokens


_CALCULATE_BOUNDS_COMMANDS = _extract_commands("calculate_bounds.py")


class TestReadmeCommands:
    """Verify that example commands in the README are valid CLI invocations."""

    @pytest.mark.parametrize("command", _CALCULATE_BOUNDS_COMMANDS)
    def test_calculate_bounds_parse_arguments(self, command):
        """Each README command for calculate_bounds.py must be parseable."""
        from scripts.calculate_bounds import parse_arguments

        argv = _command_to_argv(command)
        args = parse_arguments(argv)
        assert args is not None
