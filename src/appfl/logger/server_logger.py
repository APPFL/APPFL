import uuid
import pathlib
import logging
from typing import Optional
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


class ServerAgentFileLogger:
    """
    ServerAgentFileLogger logs FL server-side messages to the console and to a file.

    :param file_dir: The directory to save the log file.
    :param file_name: The name of the log file.
    :param experiment_id: An optional string to identify the experiment.
        If not provided, a UUID is used to ensure logger uniqueness.
    """

    def __init__(
        self,
        file_dir: str = "",
        file_name: str = "",
        experiment_id: str = "",
        prefix: str = "appfl",
    ) -> None:
        if file_name != "":
            file_name += (
                f"_Server_{experiment_id}"
                if experiment_id != ""
                else f"_Server_{uuid.uuid4().hex[:8]}"
            )

        # Unique logger name prevents collisions when multiple loggers share __name__
        logger_name = (
            __name__
            + "_"
            + (
                f"{file_dir}/{file_name}".replace("/", "_")
                if file_name
                else str(uuid.uuid4())
            )
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        colored_prefix = f"{Fore.BLUE}{Style.BRIGHT}{prefix}: "
        reset = Style.RESET_ALL
        info_fmt = _RoundAwareFormatter(
            f"{colored_prefix}✅{reset}[%(asctime)s server%(round_part)s]: %(message)s"
        )
        debug_fmt = _RoundAwareFormatter(
            f"{colored_prefix}💡{reset}[%(asctime)s server%(round_part)s]: %(message)s"
        )
        error_fmt = _RoundAwareFormatter(
            f"{colored_prefix}❌{reset}[%(asctime)s server%(round_part)s]: %(message)s"
        )
        warning_fmt = _RoundAwareFormatter(
            f"{colored_prefix}❗️{reset}[%(asctime)s server%(round_part)s]: %(message)s"
        )

        # Plain formatters for file output (no ANSI color codes)
        info_fmt_file = _RoundAwareFormatter(
            f"{prefix}: ✅[%(asctime)s server%(round_part)s]: %(message)s"
        )
        debug_fmt_file = _RoundAwareFormatter(
            f"{prefix}: 💡[%(asctime)s server%(round_part)s]: %(message)s"
        )
        error_fmt_file = _RoundAwareFormatter(
            f"{prefix}: ❌[%(asctime)s server%(round_part)s]: %(message)s"
        )
        warning_fmt_file = _RoundAwareFormatter(
            f"{prefix}: ❗️[%(asctime)s server%(round_part)s]: %(message)s"
        )

        num_s_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)]
        )
        num_f_handlers = len(
            [h for h in self.logger.handlers if isinstance(h, logging.FileHandler)]
        )

        if num_s_handlers == 0:
            for fmt, level in [
                (info_fmt, logging.INFO),
                (debug_fmt, logging.DEBUG),
                (error_fmt, logging.ERROR),
                (warning_fmt, logging.WARNING),
            ]:
                h = logging.StreamHandler()
                h.setFormatter(fmt)
                h.addFilter(LevelFilter(level))
                self.logger.addHandler(h)

        if file_dir != "" and file_name != "" and num_f_handlers == 0:
            pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
            real_file_name = f"{file_dir}/{file_name}.txt"
            self.log_filepath = real_file_name
            for fmt, level in [
                (info_fmt_file, logging.INFO),
                (debug_fmt_file, logging.DEBUG),
                (error_fmt_file, logging.ERROR),
                (warning_fmt_file, logging.WARNING),
            ]:
                h = logging.FileHandler(real_file_name)
                h.setFormatter(fmt)
                h.addFilter(LevelFilter(level))
                self.logger.addHandler(h)
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
