import os
import uuid
import logging
import pathlib
from typing import List, Dict, Union

class ClientAgentFileLogger:
    def __init__(self, logging_id: str="", file_dir: str="", file_name: str="") -> None:
        file_name += f"_{logging_id}"
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s]: %(message)s') if logging_id == "" else logging.Formatter(f'[%(asctime)s %(levelname)-4s {logging_id}]: %(message)s')
        self.logger = logging.getLogger(__name__+"_"+logging_id if logging_id != "" else str(uuid.uuid4()))
        self.logger.setLevel(logging.INFO)
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        self.logger.addHandler(s_handler)
        if file_dir != "" and file_name != "":
            if not os.path.exists(file_dir):
                pathlib.Path(file_dir).mkdir(parents=True)
            uniq = 1
            real_file_name = f"{file_dir}/{file_name}.txt"
            while os.path.exists(real_file_name):
                real_file_name = f"{file_dir}/{file_name}_{uniq}.txt"
                uniq += 1
            self.logger.info(f"Logging to {real_file_name}")
            f_handler = logging.FileHandler(real_file_name)
            f_handler.setLevel(logging.INFO)
            f_handler.setFormatter(fmt)
            self.logger.addHandler(f_handler)

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