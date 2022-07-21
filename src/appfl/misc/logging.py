import imp
import logging
from omegaconf import DictConfig
import os
from datetime import datetime
import json
class EvalLogger:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.test_results = {'clients': None, 'server': None}
        self.val_results  = {'clients': [], 'server': []}
        self.main_logger = mLogging.get_logger()

    def __format(self, key, val):
        c = "%8s:" % key
        if type(val) == int:
            c+= "%5s"    % val 
        else:
            c+= "%10.3f" % val
        return c

    def log_info_client_results(self, results, rs_name = ''):
        for cli_name in results:
            c = "[%8s] %20s" %  (rs_name, cli_name)
            for k in results[cli_name]:
                c+= self.__format(k, results[cli_name][k])
            self.main_logger.info(c)
    
    def log_info_server_results(self, results, rs_name = ''):
        c = "[%8s] %20s" % (rs_name, "server")
        for k in results:
            c+= self.__format(k, results[k])
        self.main_logger.info(c)

    def log_client_testing(self, results):
        rs = {
            self.cfg.clients[client_idx].name: results[client_idx] for client_idx in results
        }
        self.test_results['clients'] = rs
        self.log_info_client_results(rs, 'CLI-TEST')
    
    def log_server_testing(self, results):
        self.test_results['server'] = results 
        self.log_info_server_results(results, 'SER-TEST')

    def log_client_validation(self, results, step):
        rs = {
            self.cfg.clients[client_idx].name: {
                'step': step, **results[client_idx],
                } for client_idx in results
        }
        self.val_results['clients'].append(rs)
        self.log_info_client_results(rs, 'CLI-VAL')
    
    def log_server_validation(self, results, step):
        rs = { 'step': step, **results }
        self.val_results['server'].append(rs)
        self.log_info_server_results(rs, 'SER-VAL')
    
    def save_log(self, output_file):
        out_dict = {"val": self.val_results, "test": self.test_results}
        with open(output_file, "w") as fo:
            json.dump(out_dict, fo)

class mLogging:
    __logger = None
    @classmethod
    def config_logger(cls, cfg: DictConfig):
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

        # Instantiate method
        new_inst = cls.__new__(cls) 
        new_inst.logger    = logger
        new_inst.dir       = dir
        new_inst.timestamp = time_stamp
        if cfg.use_tensorboard:
            tb_dir = os.path.join(
                dir,
                "tensorboard",
                "%s_%s" % (run_str, time_stamp)
            )
            from tensorboardX import SummaryWriter
            new_inst.writer = SummaryWriter(tb_dir)
        
        # Initialize eval logger
        cls.__logger = new_inst
        new_inst.eval_logger= EvalLogger(cfg)
        

    @classmethod
    def get_logger(cls):
        if cls.__logger is None:
            raise RuntimeError("Need to configure logger first")
        return cls.__logger.logger
    
    @classmethod
    def get_eval_logger(cls):
        if cls.__logger is None:
            raise RuntimeError("Need to configure logger first")
        return cls.__logger.eval_logger
    
    @classmethod
    def get_tensorboard_writer(cls):
        if cls.__logger.writer is None:
            raise Exception("Tensorboard X writer need to be configured first")
        return cls.__logger.writer

    @classmethod
    def save_funcx_log(cls, cfg):
        import csv
        header = ['timestamp','task_name','client_name','status','execution_time']
        lgg = cls.__logger
        with open(os.path.join(
                lgg.dir, "log_funcx_%s.csv" % lgg.timestamp
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
        # Save log
        lgg.eval_logger.save_log(os.path.join(lgg.dir, "log_eval_%s.json" % lgg.timestamp))
