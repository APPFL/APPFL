"""Console + file logger for the virtual-time simulator.

Matches the look of APPFL's own loggers: the coloured ``appfl: ✅`` prefix and
one emoji per level (``appfl.logger.ServerAgentFileLogger``), plus the aligned
column helpers that client trainers use for their metric tables
(``appfl.logger.ClientAgentFileLogger``). Simulator events therefore stream as
a readable table rather than ad-hoc f-strings.

Two deliberate differences from the APPFL loggers:

- Colour codes are applied only to the console handlers. The log file gets a
  plain formatter, so it stays greppable instead of carrying ANSI escapes.
- The log file is named verbatim from ``file_name`` (no appended timestamp),
  because the simulator already timestamps its per-run output directory.
"""

import logging
import os
import pathlib
from typing import Any

from colorama import Fore, Style

from appfl.logger.utils import LevelFilter

# Emoji per level, matching appfl.logger
_LEVEL_EMOJI = {
    logging.INFO: "✅",
    logging.DEBUG: "💡",
    logging.ERROR: "❌",
    logging.WARNING: "❗️",
}

_MIN_COL_WIDTH = 10
_MIN_RULE_WIDTH = 72


class _TableMixin:
    """Aligned-column helpers, mirroring ``ClientAgentFileLogger``."""

    def log_title(self, titles: list, repeat: bool = False) -> None:
        """
        Record the column headers for subsequent :meth:`log_content` calls.

        :param titles: Column names, right-aligned to the shared column width.
        :param repeat: When True the header is reprinted immediately above every
            row instead of once here. Simulator events interleave with APPFL's
            own per-client training tables, so a header printed once ends up
            scrolled far away from the rows it labels.
        """
        self.titles = titles
        self._repeat_title = repeat
        if not repeat:
            self.info(self._title_row())

    def set_title(self, titles: list) -> None:
        """Record column headers without printing them."""
        if not hasattr(self, "titles"):
            self.titles = titles

    def _title_row(self) -> str:
        """Render the header row for the current titles."""
        return " ".join(["%10s" % t for t in self.titles])

    def log_content(self, contents: dict | list) -> None:
        """
        Print one row under the headers set by :meth:`log_title`.

        A dict may omit columns; the missing ones render as blanks, which lets
        event kinds with different fields share a single table.
        """
        if not isinstance(contents, (dict, list)):
            raise ValueError("Contents must be a dictionary or list")
        if isinstance(contents, dict):
            for key in contents:
                if key not in self.titles:
                    raise ValueError(f"Title {key} is not defined")
            contents = [contents.get(key, "") for key in self.titles]
        elif len(contents) != len(self.titles):
            raise ValueError("Contents and titles must have the same length")
        widths = [max(len(str(t)), _MIN_COL_WIDTH) for t in self.titles]
        row = " ".join(
            [
                "%*.4f" % (w, c) if isinstance(c, float) else "%*s" % (w, c)
                for w, c in zip(widths, contents)
            ]
        )
        if getattr(self, "_repeat_title", False):
            self.info(self._title_row())
        # Blank columns pad to full width; trim so partial rows end cleanly.
        self.info(row.rstrip())

    def log_banner(self, title: str, fields: dict[str, Any] | None = None) -> None:
        """Print a rule, a title, and an optional row of ``key=value`` fields."""
        lines = [title]
        if fields:
            lines.append("  ".join(f"{k}={v}" for k, v in fields.items()))
        rule = "─" * max(_MIN_RULE_WIDTH, max(len(line) for line in lines))
        self.info(rule)
        for line in lines:
            self.info(line)
        self.info(rule)


class VsimLogger(_TableMixin):
    """
    Logs virtual-time simulation messages to the console and, optionally, a file.

    :param logging_id: Label shown in the log prefix (defaults to ``vsim``).
    :param file_dir: Directory for the log file; empty disables file logging.
    :param file_name: Base name of the log file, used verbatim with a ``.log``
        suffix.
    """

    def __init__(
        self,
        logging_id: str = "vsim",
        file_dir: str = "",
        file_name: str = "",
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{logging_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        for level, emoji in _LEVEL_EMOJI.items():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    f"{Fore.BLUE}{Style.BRIGHT}appfl: {emoji}{Style.RESET_ALL}"
                    f"[%(asctime)s {logging_id}]: %(message)s",
                    "%H:%M:%S",
                )
            )
            handler.addFilter(LevelFilter(level))
            self.logger.addHandler(handler)

        if file_dir != "" and file_name != "":
            pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
            self.log_filepath = os.path.join(file_dir, f"{file_name}.log")
            file_handler = logging.FileHandler(self.log_filepath)
            file_handler.setFormatter(
                logging.Formatter(
                    f"[%(asctime)s {logging_id}] %(levelname)-7s %(message)s",
                    "%H:%M:%S",
                )
            )
            self.logger.addHandler(file_handler)

    def info(self, info: str) -> None:
        self.logger.info(info)

    def debug(self, debug: str) -> None:
        self.logger.debug(debug)

    def error(self, error: str) -> None:
        self.logger.error(error)

    def warning(self, warning: str) -> None:
        self.logger.warning(warning)

    def get_log_filepath(self) -> str | None:
        return getattr(self, "log_filepath", None)


class _PlainTableAdapter(_TableMixin):
    """Adds the table helpers to a plain stdlib logger, changing nothing else."""

    def __init__(self, logger):
        self._logger = logger

    def info(self, info: str) -> None:
        self._logger.info(info)

    def debug(self, debug: str) -> None:
        self._logger.debug(debug)

    def error(self, error: str) -> None:
        self._logger.error(error)

    def warning(self, warning: str) -> None:
        self._logger.warning(warning)


def ensure_table_logger(logger):
    """
    Return a logger that supports the table helpers.

    Passes :class:`VsimLogger` through untouched, and wraps a plain stdlib
    logger so callers can still hand one in (as the unit tests do).
    """
    return logger if hasattr(logger, "log_content") else _PlainTableAdapter(logger)
