"""Set of utilities used throughout the package."""

import argparse
import logging
import os
import signal
import sys
from enum import Enum
from pathlib import Path

LOGGER = logging.getLogger(__name__)
LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
DIR_DATASETS = Path("../data").resolve()


class EnumAction(argparse.Action):
    """An argparse action to handle Enum types by name (not value)."""

    def __init__(self, *_, **kwargs):
        # Get the Enum type
        enum_type = kwargs.pop("type", None)
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum based on name (not value)
        kwargs.setdefault("choices", tuple(e.name for e in enum_type))

        # Continue usual workflow
        super().__init__(**kwargs)

        # Keep track of Enum type
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        if isinstance(values, str):
            enum_values = self._enum[values]
            setattr(namespace, self.dest, enum_values)
        elif isinstance(values, list):
            enum_values = [self._enum[v] for v in values]
            setattr(namespace, self.dest, enum_values)
        elif values is None:
            raise argparse.ArgumentTypeError(
                f"You need to pass a value after '{option_string}'"
            )
        else:
            raise argparse.ArgumentTypeError(
                f"EnumAction value type not supported: {type(values)}"
            )


def get_logging_levels():
    """Returns a list of logging levels"""
    return [logging._levelToName[v] for v in sorted(logging._levelToName)]


def _install_exception_hooks():
    """Route uncaught exceptions, unraisable exceptions, signals, and
    KeyboardInterrupt through the logging framework so they appear in log files."""
    logger = logging.getLogger(__name__)

    def excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            # Catch and log
            logger.critical("KeyboardInterrupt", exc_info=(exc_type, exc_value, exc_tb))
            logging.shutdown()

            # Resume default behavior (print to terminal, exit with code 130)
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            sys.exit(130)

        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    def unraisablehook(unraisable):
        logger.warning(
            "Unraisable exception in %s: %s",
            unraisable.object,
            unraisable.exc_value,
            exc_info=(
                type(unraisable.exc_value),
                unraisable.exc_value,
                unraisable.exc_tb,
            )
            if unraisable.exc_value
            else None,
        )

    def handle_signal(signum, _frame):
        sig_name = signal.Signals(signum).name
        logger.critical("Script terminated by signal %s (%d)", sig_name, signum)
        logging.shutdown()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    sys.excepthook = excepthook
    sys.unraisablehook = unraisablehook
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGABRT, handle_signal)


def setup_logging(level: str, fmt: str = LOGGING_FORMAT, path_log: Path = None):
    """Sets up the logging configuration"""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Create formatter
    formatter = logging.Formatter(fmt)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if path_log is provided)
    if path_log:
        file_handler = logging.FileHandler(str(path_log))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Route uncaught and unraisable exceptions through the logging framework
        # so they appear in the log file, not just on stderr
        _install_exception_hooks()

    # Capture warnings inside our logs
    logging.captureWarnings(True)

    # Disable some warnings from conflicting libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
