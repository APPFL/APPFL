import os
import uuid
import logging
import pathlib
from typing import List, Dict, Union
from .utils import LevelFilter, _RoundAwareFormatter

try:
    from colorama import Fore, Style
except Exception:  # pragma: no cover

    class _ColorStub:
        BLUE = ""
        BRIGHT = ""
        RESET_ALL = ""

    Fore = _ColorStub()
    Style = _ColorStub()


class ClientAgentFileLogger:
    """
    ClientAgentFileLogger logs FL client-side messages to the console and to a file.

    :param logging_id: An optional string to identify the client.
    :param file_dir: The directory to save the log file.
    :param file_name: The name of the log file.
    :param experiment_id: An optional string to identify the experiment.
    :param title_every_n: Re-print the column header every N content rows (0 = never repeat).
    :param show_titles: Whether to print column headers at all.
    """

    def __init__(
        self,
        logging_id: str = "",
        file_dir: str = "",
        file_name: str = "",
        experiment_id: str = "",
        title_every_n: int = 20,
        show_titles: bool = True,
    ) -> None:
        self.title_every_n = int(title_every_n)
        self.show_titles = bool(show_titles)
        self._content_count = 0
        self._widths: List[int] = []
        self._round_label = ""

        if file_name != "":
            file_name += f"_{logging_id}" if logging_id != "" else ""
            file_name += (
                f"_{experiment_id if experiment_id != '' else uuid.uuid4().hex[:8]}"
            )

        if logging_id == "":
            client_label = "Client"
        else:
            client_label = f"Client {logging_id}"

        # Unique logger name prevents collisions across multiple client loggers
        logger_name = (
            __name__
            + "_"
            + (
                f"{file_dir}/{file_name}".replace("/", "_")
                if file_name
                else (logging_id if logging_id != "" else str(uuid.uuid4()))
            )
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        prefix = f"{Fore.BLUE}{Style.BRIGHT}appfl: "
        reset = Style.RESET_ALL

        def _make_fmt(icon: str, colored: bool) -> _RoundAwareFormatter:
            if colored:
                if logging_id == "":
                    pat = f"{prefix}{icon}{reset}[%(asctime)s]: %(message)s"
                else:
                    pat = f"{prefix}{icon}{reset}[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            else:
                if logging_id == "":
                    pat = f"appfl: {icon}[%(asctime)s]: %(message)s"
                else:
                    pat = f"appfl: {icon}[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            return _RoundAwareFormatter(pat)

        icons = {"info": "✅", "debug": "💡", "error": "❌", "warning": "❗️"}
        levels = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "error": logging.ERROR,
            "warning": logging.WARNING,
        }

        num_s_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)]
        )
        num_f_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.FileHandler)]
        )

        if num_s_handlers == 0:
            for key, level in levels.items():
                h = logging.StreamHandler()
                h.setFormatter(_make_fmt(icons[key], colored=True))
                h.addFilter(LevelFilter(level))
                self.logger.addHandler(h)

        if file_dir != "" and file_name != "" and num_f_handlers == 0:
            pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
            real_file_name = f"{file_dir}/{file_name}.txt"
            file_exists = os.path.exists(real_file_name)
            for key, level in levels.items():
                h = logging.FileHandler(real_file_name)
                h.setFormatter(_make_fmt(icons[key], colored=False))
                h.addFilter(LevelFilter(level))
                self.logger.addHandler(h)
            if not file_exists:
                self.info(f"Logging to {real_file_name}")

    def log_title(self, titles: List) -> None:
        self.titles = titles
        self._widths = [max(len(str(t)), 10) for t in titles]

    def set_title(self, titles: List) -> None:
        if not hasattr(self, "titles"):
            self.titles = titles

    def set_round_label(self, round_label: str) -> None:
        self._round_label = str(round_label)

    def log_content(self, contents: Union[Dict, List]) -> None:
        if not isinstance(contents, (dict, list)):
            raise ValueError("Contents must be a dictionary or list")
        if not isinstance(contents, list):
            for key in contents.keys():
                if key not in self.titles:
                    raise ValueError(f"Title {key} is not defined")
            contents = [contents.get(key, "") for key in self.titles]
        else:
            if len(contents) != len(self.titles):
                raise ValueError("Contents and titles must have the same length")
        if not self._widths:
            self._widths = [max(len(str(t)), 10) for t in self.titles]
        if (
            self.show_titles
            and self.title_every_n > 0
            and self._content_count % self.title_every_n == 0
        ):
            header = " ".join(
                ["%*s" % (w, t) for w, t in zip(self._widths, self.titles)]
            )
            self.info(header)
        content = " ".join(
            [
                "%*s" % (w, c) if not isinstance(c, float) else "%*.4f" % (w, c)
                for w, c in zip(self._widths, contents)
            ]
        )
        self.info(content, round_label=self._round_label)
        self._content_count += 1

    def info(self, info: str, round_label: str = "") -> None:
        label = round_label if round_label else self._round_label
        self.logger.info(info, extra={"round_label": label})

    def debug(self, debug: str, round_label: str = "") -> None:
        label = round_label if round_label else self._round_label
        self.logger.debug(debug, extra={"round_label": label})

    def error(self, error: str, round_label: str = "") -> None:
        label = round_label if round_label else self._round_label
        self.logger.error(error, extra={"round_label": label})

    def warning(self, warning: str, round_label: str = "") -> None:
        label = round_label if round_label else self._round_label
        self.logger.warning(warning, extra={"round_label": label})
