import os
import pathlib
import logging
from datetime import datetime

class ServerAgentFileLogger:
    """
    ServerAgentFileLogger is a class that logs FL server-side messages to the console and to a file.
    :param file_dir: The directory to save the log file.
    :param file_name: The name of the log file.
    :param experiment_id: An optional string to identify the experiment. 
        If not provided, the current date and time will be used.
    """
    def __init__(
        self, 
        file_dir: str="", 
        file_name: str="", 
        experiment_id: str=""
    ) -> None:
        if file_name != "":
            file_name += f"_Server_{experiment_id if experiment_id != '' else datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        num_s_handlers = len([h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)])
        num_f_handlers = len([h for h in self.logger.handlers if isinstance(h, logging.FileHandler)])

        if num_s_handlers == 0:
            s_handler = logging.StreamHandler()
            s_handler.setLevel(logging.INFO)
            s_handler.setFormatter(fmt)
            self.logger.addHandler(s_handler)
        if file_dir != "" and file_name != "" and num_f_handlers == 0:
            if not os.path.exists(file_dir):
                pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)
            real_file_name = f"{file_dir}/{file_name}.txt"
            f_handler = logging.FileHandler(real_file_name)
            f_handler.setLevel(logging.INFO)
            f_handler.setFormatter(fmt)
            self.logger.addHandler(f_handler)
            self.logger.info(f"Logging to {real_file_name}")

    def info(self, info: str) -> None:
        self.logger.info(info)

    def debug(self, debug: str) -> None:
        self.logger.debug(debug)

    def error(self, error: str) -> None:
        self.logger.error(error)