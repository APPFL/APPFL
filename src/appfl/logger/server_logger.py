import os
import pathlib
import logging
from datetime import datetime

class ServerAgentFileLogger:
    def __init__(self, file_dir: str="", file_name: str="", experiment_id: str="") -> None:
        if file_name != "":
            file_name += f"_Server_{experiment_id if experiment_id != '' else datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        self.logger.addHandler(s_handler)
        if file_dir != "" and file_name != "":
            if not os.path.exists(file_dir):
                pathlib.Path(file_dir).mkdir(parents=True)
            real_file_name = f"{file_dir}/{file_name}.txt"
            self.logger.info(f"Logging to {real_file_name}")
            f_handler = logging.FileHandler(real_file_name)
            f_handler.setLevel(logging.INFO)
            f_handler.setFormatter(fmt)
            self.logger.addHandler(f_handler)

    def info(self, info: str) -> None:
        self.logger.info(info)

    def debug(self, debug: str) -> None:
        self.logger.debug(debug)

    def error(self, error: str) -> None:
        self.logger.error(error)