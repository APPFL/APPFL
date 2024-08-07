import os
import uuid
import logging
import pathlib
from datetime import datetime
from typing import List, Dict, Union

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
        logging_id: str="", 
        file_dir: str="", 
        file_name: str="", 
        experiment_id: str=""
    ) -> None:
        
        if file_name != "":
            file_name += f"_{logging_id}" if logging_id != "" else ""
            file_name += f"_{experiment_id if experiment_id != '' else datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        fmt = (
            logging.Formatter('[%(asctime)s %(levelname)-4s]: %(message)s') 
            if logging_id == "" 
            else logging.Formatter(f'[%(asctime)s %(levelname)-4s {logging_id}]: %(message)s')
        )
        self.logger = logging.getLogger(__name__+"_"+logging_id if logging_id != "" else str(uuid.uuid4()))
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

    def log_title(self, titles: List) -> None:
        self.titles = titles
        title = " ".join(["%10s" % t for t in titles])
        self.logger.info(title)
    
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
        length = [max(len(str(t)), 10) for t in self.titles]
        content = " ".join(["%*s" % (l, c) if not isinstance(c, float) else "%*.4f" % (l, c) for l, c in zip(length, contents)])
        self.logger.info(content)

    def info(self, info: str) -> None:
        self.logger.info(info)

    def debug(self, debug: str) -> None:
        self.logger.debug(debug)

    def error(self, error: str) -> None:
        self.logger.error(error)