import os
import pathlib
import logging

class ServerAgentFileLogger:
    def __init__(self, file_dir: str="", file_name: str="") -> None:
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        file_name += "_server"
        self.logger = logging.getLogger(__name__)
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

    def info(self, info: str) -> None:
        self.logger.info(info)