import os
import uuid
import logging
import pathlib
from appfl.logger.utils import LevelFilter, _RoundAwareFormatter
from typing import List, Dict, Union

try:
    from colorama import Fore, Style
except Exception:  # pragma: no cover
    class _ColorStub:
        BLUE = ''
        BRIGHT = ''
        RESET_ALL = ''

    Fore = _ColorStub()
    Style = _ColorStub()


class ClientAgentFileLogger:
    """
    ClientAgentFileLogger is a class that logs FL client-side messages to the console and to a file.
    :param logging_id: An optional string to identify the client.
    :param file_dir: The directory to save the log file.
    :param file_name: The name of the log file.
    :param experiment_id: An optional string to identify the experiment.
        If not provided, the current date and time will be used.
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
        del experiment_id
        self.title_every_n = int(title_every_n)
        self.show_titles = bool(show_titles)
        self._content_count = 0
        self._widths: List[int] = []
        self._round_label = ""
        if file_name != "":
            if logging_id != "":
                try:
                    file_name = f"{file_name}_{int(logging_id):04d}"
                except Exception:
                    file_name = f"{file_name}_{logging_id}"
            if not file_name.endswith(".log"):
                file_name = f"{file_name}.log"

        if logging_id == "":
            client_label = "Client"
        else:
            try:
                client_label = f"Client {int(logging_id):04d}"
            except Exception:
                client_label = f"Client {logging_id}"

        logger_name = __name__ + "_" + (
            f"{file_dir}/{file_name}".replace("/", "_")
            if file_name
            else (logging_id if logging_id != "" else str(uuid.uuid4()))
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        info_fmt = (
            _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ✅{Style.RESET_ALL}[%(asctime)s]: %(message)s"
            )
            if logging_id == ""
            else _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ✅{Style.RESET_ALL}[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )
        debug_fmt = (
            _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: 💡{Style.RESET_ALL}[%(asctime)s]: %(message)s"
            )
            if logging_id == ""
            else _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: 💡{Style.RESET_ALL}[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )
        error_fmt = (
            _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ❌{Style.RESET_ALL}[%(asctime)s]: %(message)s"
            )
            if logging_id == ""
            else _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ❌{Style.RESET_ALL}[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )
        warning_fmt = (
            _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ❗️{Style.RESET_ALL}[%(asctime)s]: %(message)s"
            )
            if logging_id == ""
            else _RoundAwareFormatter(
                f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ❗️{Style.RESET_ALL}[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )
        info_fmt_file = (
            _RoundAwareFormatter("appfl-sim: ✅[%(asctime)s]: %(message)s")
            if logging_id == ""
            else _RoundAwareFormatter(
                f"appfl-sim: ✅[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )
        debug_fmt_file = (
            _RoundAwareFormatter("appfl-sim: 💡[%(asctime)s]: %(message)s")
            if logging_id == ""
            else _RoundAwareFormatter(
                f"appfl-sim: 💡[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )
        error_fmt_file = (
            _RoundAwareFormatter("appfl-sim: ❌[%(asctime)s]: %(message)s")
            if logging_id == ""
            else _RoundAwareFormatter(
                f"appfl-sim: ❌[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )
        warning_fmt_file = (
            _RoundAwareFormatter("appfl-sim: ❗️[%(asctime)s]: %(message)s")
            if logging_id == ""
            else _RoundAwareFormatter(
                f"appfl-sim: ❗️[%(asctime)s | {client_label}%(round_part)s]: %(message)s"
            )
        )

        num_s_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)]
        )
        num_f_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.FileHandler)]
        )

        if num_s_handlers == 0:
            s_handler_info = logging.StreamHandler()
            s_handler_info.setFormatter(info_fmt)
            s_handler_info.addFilter(LevelFilter(logging.INFO))
            s_handler_debug = logging.StreamHandler()
            s_handler_debug.setFormatter(debug_fmt)
            s_handler_debug.addFilter(LevelFilter(logging.DEBUG))
            s_handler_error = logging.StreamHandler()
            s_handler_error.setFormatter(error_fmt)
            s_handler_error.addFilter(LevelFilter(logging.ERROR))
            s_handler_warning = logging.StreamHandler()
            s_handler_warning.setFormatter(warning_fmt)
            s_handler_warning.addFilter(LevelFilter(logging.WARNING))
            self.logger.addHandler(s_handler_info)
            self.logger.addHandler(s_handler_debug)
            self.logger.addHandler(s_handler_error)
            self.logger.addHandler(s_handler_warning)

        if file_dir != "" and file_name != "" and num_f_handlers == 0:
            if not os.path.exists(file_dir):
                pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
            real_file_name = f"{file_dir}/{file_name}"
            # check if the file exists
            file_exists = os.path.exists(real_file_name)
            f_handler_info = logging.FileHandler(real_file_name)
            f_handler_info.setFormatter(info_fmt_file)
            f_handler_info.addFilter(LevelFilter(logging.INFO))
            f_handler_debug = logging.FileHandler(real_file_name)
            f_handler_debug.setFormatter(debug_fmt_file)
            f_handler_debug.addFilter(LevelFilter(logging.DEBUG))
            f_handler_error = logging.FileHandler(real_file_name)
            f_handler_error.setFormatter(error_fmt_file)
            f_handler_error.addFilter(LevelFilter(logging.ERROR))
            f_handler_warning = logging.FileHandler(real_file_name)
            f_handler_warning.setFormatter(warning_fmt_file)
            f_handler_warning.addFilter(LevelFilter(logging.WARNING))
            self.logger.addHandler(f_handler_info)
            self.logger.addHandler(f_handler_debug)
            self.logger.addHandler(f_handler_error)
            self.logger.addHandler(f_handler_warning)
            if not file_exists:
                self.info(f"Logging to {real_file_name}")

    def log_title(self, titles: List) -> None:
        self.titles = titles
        self._widths = [max(len(str(t)), 10) for t in titles]
        # Do not emit an unconditional header line here.
        # Headers are emitted by `log_content` according to `title_every_n`.

    def log_content(self, contents: Union[Dict, List]) -> None:
        if not isinstance(contents, dict) and not isinstance(contents, list):
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
                ["%*s" % (ln, t) for ln, t in zip(self._widths, self.titles)]
            )
            self.info(header)
        content = " ".join(
            [
                "%*s" % (ln, cnt) if not isinstance(cnt, float) else "%*.4f" % (ln, cnt)
                for ln, cnt in zip(self._widths, contents)
            ]
        )
        self.info(content, round_label=self._round_label)
        self._content_count += 1

    def set_round_label(self, round_label: str) -> None:
        self._round_label = str(round_label)

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
