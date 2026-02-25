import os
import pathlib
import logging
import uuid
from appfl.logger.utils import LevelFilter, _RoundAwareFormatter
from typing import Optional
try:
    from colorama import Fore, Style
except Exception:  # pragma: no cover
    class _ColorStub:
        BLUE = ''
        BRIGHT = ''
        RESET_ALL = ''

    Fore = _ColorStub()
    Style = _ColorStub()


class ServerAgentFileLogger:
    """
    ServerAgentFileLogger is a class that logs FL server-side messages to the console and to a file.
    :param file_dir: The directory to save the log file.
    :param file_name: The name of the log file.
    :param experiment_id: An optional string to identify the experiment.
        If not provided, the current date and time will be used.
    """

    def __init__(
        self, file_dir: str = "", file_name: str = "", experiment_id: str = ""
    ) -> None:
        del experiment_id
        if file_name != "" and not file_name.endswith(".log"):
            file_name = f"{file_name}.log"
        logger_name = __name__ + "_" + (
            f"{file_dir}/{file_name}".replace("/", "_") if file_name else str(uuid.uuid4())
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        info_fmt = _RoundAwareFormatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ✅{Style.RESET_ALL}[%(asctime)s | Server%(round_part)s]: %(message)s"
        )
        debug_fmt = _RoundAwareFormatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: 💡{Style.RESET_ALL}[%(asctime)s | Server%(round_part)s]: %(message)s"
        )
        error_fmt = _RoundAwareFormatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ❌{Style.RESET_ALL}[%(asctime)s | Server%(round_part)s]: %(message)s"
        )
        warning_fmt = _RoundAwareFormatter(
            f"{Fore.BLUE}{Style.BRIGHT}appfl-sim: ❗️{Style.RESET_ALL}[%(asctime)s | Server%(round_part)s]: %(message)s"
        )
        info_fmt_file = _RoundAwareFormatter(
            "appfl-sim: ✅[%(asctime)s | Server%(round_part)s]: %(message)s"
        )
        debug_fmt_file = _RoundAwareFormatter(
            "appfl-sim: 💡[%(asctime)s | Server%(round_part)s]: %(message)s"
        )
        error_fmt_file = _RoundAwareFormatter(
            "appfl-sim: ❌[%(asctime)s | Server%(round_part)s]: %(message)s"
        )
        warning_fmt_file = _RoundAwareFormatter(
            "appfl-sim: ❗️[%(asctime)s | Server%(round_part)s]: %(message)s"
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
            self.log_filepath = real_file_name
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
            self.info(f"Logging to {real_file_name}")

    def info(self, info: str, round_label: str = "") -> None:
        self.logger.info(info, extra={"round_label": round_label})

    def debug(self, debug: str, round_label: str = "") -> None:
        self.logger.debug(debug, extra={"round_label": round_label})

    def error(self, error: str, round_label: str = "") -> None:
        self.logger.error(error, extra={"round_label": round_label})

    def warning(self, warning: str, round_label: str = "") -> None:
        self.logger.warning(warning, extra={"round_label": round_label})

    def get_log_filepath(self) -> Optional[str]:
        if hasattr(self, "log_filepath"):
            return self.log_filepath
