import imp
import logging
from omegaconf import DictConfig
import os
from datetime import datetime

class mLogging:
    __logger    = None
    __writer    = None
    __dir       = None 
    __timestamp = None
    @staticmethod
    def config_logger(cfg: DictConfig):
        run_str = "%s_%s_%s" % (cfg.dataset, cfg.fed.servername, cfg.fed.args.optim)
        dir     = os.path.join(cfg.server.output_dir,
                                "outputs_%s" % run_str)
        
        if os.path.isdir(dir) == False:
            os.makedirs(dir, exist_ok = True)
        
        
        time_stamp = datetime.now().strftime("%m%d%y_%H%M%S")
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s]: %(message)s') 
        log_fname  = os.path.join(
            dir,
            "log_server_%s.log" % time_stamp) 
    
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_fname)
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        c_handler.setFormatter(fmt)
        f_handler.setFormatter(fmt)
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        mLogging.__logger    = logger
        mLogging.__dir       = dir
        mLogging.__timestamp = time_stamp
        if cfg.use_tensorboard:
            tb_dir = os.path.join(
                dir,
                "tensorboard",
                "%s_%s" % (run_str, time_stamp)
            )
            from tensorboardX import SummaryWriter
            mLogging.__writer = SummaryWriter(tb_dir)

    @staticmethod
    def get_logger():
        if mLogging.__logger is None:
            raise Exception("Logger need to be configured first")
        return mLogging.__logger

    @staticmethod
    def get_tensorboard_writer():
        if mLogging.__writer is None:
            raise Exception("Tensorboard X writer need to be configured first")
        return mLogging.__writer

    @staticmethod
    def save_funcx_log(cfg):
        import csv
        header = ['timestamp','task_name','client_name','status','execution_time']
        
        with open(os.path.join(
                mLogging.__dir,
                "log_funcx_%s.csv" % mLogging.__timestamp
            ), "w") as fo:
            writer = csv.writer(fo)
            writer.writerow(header)
            for i, tlog in enumerate(cfg.logging_tasks):
                writer.writerow([
                    datetime.fromtimestamp(tlog.start_time),
                    tlog.task_name,
                    cfg.clients[tlog.client_idx].name,
                    "success" if tlog.success else "failed",        
                    "%.02f" % (tlog.end_time - tlog.start_time)    
                ])
