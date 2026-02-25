import logging


class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        if record.levelno == self.level:
            return True
        # Keep CRITICAL records visible via the ERROR handler.
        return self.level == logging.ERROR and record.levelno == logging.CRITICAL
