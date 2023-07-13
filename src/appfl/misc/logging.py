import json
import logging
import os
import time
import torch

from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import os.path as osp

TIME_STR = "%m%d%y_%H%M%S"

class EvalLogger:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.test_results = {"clients": None, "server": None}
        self.val_results = {"clients": [], "server": []}
        self.main_logger = mLogging.get_logger()

    def __format(self, key, val):
        if type(val) == list:
            return ""
        c = "%10s:" % key
        if type(val) == int:
            c += "%5s" % val
        elif type:
            c += "%10.4f" % val
        return c

    def log_info_client_results(self, results, rs_name=""):
        for cli_name in results:
            c = "[%8s] %20s" % (rs_name, cli_name)
            for k in results[cli_name]:
                c += self.__format(k, results[cli_name][k])
            self.main_logger.info(c)

    def log_info_server_results(self, results, rs_name=""):
        c = "[%8s] %20s" % (rs_name, "server")
        for k in results:
            c += self.__format(k, results[k])
        self.main_logger.info(c)

    def log_client_testing(self, results):
        rs = {
            self.cfg.clients[client_idx].name: results[client_idx]
            for client_idx in results
        }
        self.test_results["clients"] = rs
        self.log_info_client_results(rs, "CLI-TEST")

    def log_server_testing(self, results):
        self.test_results["server"] = results
        self.log_info_server_results(results, "SER-TEST")

    def log_client_validation(self, results, step):
        rs = {
            self.cfg.clients[client_idx].name: {
                "step": step,
                **results[client_idx],
            }
            for client_idx in results
        }
        self.val_results["clients"].append(rs)
        self.log_info_client_results(rs, "CLI-VAL")

    def log_server_validation(self, results, step):
        rs = {"step": step, **results}
        self.val_results["server"].append(rs)
        self.log_info_server_results(rs, "SER-VAL")

    def save_log(self, output_file):
        out_dict = {}
        if self.cfg.load_model == True:
            out_dict["checkpoint_dirname"] = self.cfg.load_model_dirname
            out_dict["checkpoint_filename"] = self.cfg.load_model_filename

        out_dict["val"] = self.val_results
        out_dict["test"] = self.test_results
        
        with open(output_file, "w") as fo:
            json.dump(out_dict, fo, indent=2)

class mLogging:
    __logger = None

    @classmethod
    def config_logger(
        cls,
        cfg: DictConfig,
        cfg_file_name=None,
        client_cfg_file_name=None,
        mode="train",
    ):
        run_str = "" #"%s_%s_%s" % (cfg.dataset, cfg.fed.servername, cfg.fed.args.optim)
        if cfg_file_name is not None and client_cfg_file_name is not None:
            run_str = "%s_%s" % (
                # run_str,
                cfg_file_name.replace(".yaml", ""),
                client_cfg_file_name.replace(".yaml", ""),
            )
        
        mode_prefix = {
            "train": "outputs", 
            "clients_testing" : "eval", 
            "attack": "attack",
            "clients_adapt_test": "adapt_test"

        }
        
        dir = os.path.join(
            cfg.server.output_dir,
            "%s_%s" % (mode_prefix[mode], run_str),
        )

        cfg.server.output_dir = dir
        if os.path.isdir(dir) == False:
            os.makedirs(dir, exist_ok=True)

        time_stamp = datetime.now().strftime(TIME_STR)
        fmt = logging.Formatter("[%(asctime)s %(levelname)-4s]: %(message)s")
        log_fname = os.path.join(dir, "log_server_%s.log" % time_stamp)

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
        new_inst.logger = logger
        new_inst.dir = dir
        new_inst.timestamp = time_stamp
        if cfg.use_tensorboard:
            tb_dir = os.path.join(dir, "tensorboard", "%s_%s" % (run_str, time_stamp))
            from tensorboardX import SummaryWriter

            new_inst.writer = SummaryWriter(tb_dir)

        # Initialize eval logger
        cls.__logger = new_inst
        new_inst.eval_logger = EvalLogger(cfg)

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
    def save_data_stats(cls, cfg, data_stats_at_client):
        if data_stats_at_client is None:
            return
        data_stats = {}
        for client_idx in range(cfg.num_clients):
            data_stats[cfg.clients[client_idx].name] = data_stats_at_client[client_idx]
        logger = cls.get_logger()
        logger.info(data_stats)
        with open(os.path.join(cls.__logger.dir, "data_stats.json"), "w") as fo:
            json.dump(data_stats, fo, indent=2)

    @classmethod
    def save_funcx_log(cls, cfg):
        import csv

        header = ["timestamp", "task_name", "client_name", "status", "execution_time"]
        lgg = cls.__logger
        # Save csv file
        with open(os.path.join(lgg.dir, "log_funcx_%s.csv" % lgg.timestamp), "w") as fo:
            writer = csv.writer(fo)
            writer.writerow(header)
            for i, tlog in enumerate(cfg.logging_tasks):
                writer.writerow(
                    [
                        datetime.fromtimestamp(tlog.start_time),
                        tlog.task_name,
                        cfg.clients[tlog.client_idx].name,
                        "success" if tlog.success else "failed",
                        "%.02f" % (tlog.end_time - tlog.start_time),
                    ]
                )

        # Save json file
        with open(
            os.path.join(lgg.dir, "log_funcx_%s.yaml" % lgg.timestamp), "w"
        ) as fo:
            log_tasks = []
            for i, tlog in enumerate(cfg.logging_tasks):
                l_tsk = {}
                l_tsk["task_name"] = tlog.task_name
                l_tsk["endpoint"] = cfg.clients[tlog.client_idx].name
                l_tsk["start_at"] = str(datetime.fromtimestamp(tlog.start_time))
                l_tsk["end_at"] = str(datetime.fromtimestamp(tlog.end_time))
                l_tsk["events"] = dict(tlog.log.events) if tlog.log is not None else None
                l_tsk["timing"] = dict(tlog.log.timing) if tlog.log is not None else None
                log_tasks.append(l_tsk)
            fo.write(OmegaConf.to_yaml(log_tasks))
        out_json_file = os.path.join(lgg.dir, "log_eval_%s_%s.json" % (
                        osp.basename(cfg.load_model_dirname),
                        cfg.load_model_filename,
                        # lgg.timestamp
                        ))
        cls.get_logger().info("Saving evaluation results to %s" % out_json_file)
        # Save eval log
        lgg.eval_logger.save_log(
            out_json_file
        )

    @classmethod
    def log_client_data_info(cls, cfg, data_info_at_client):
        mode = list(data_info_at_client[0].keys())
        logger = cls.get_logger()
        b = "|" + "-" * 25 + "|" + ("-" * 10 + "|") * len(mode)
        logger.info(b)
        c = "|%25s|" + "%10s|" * len(mode)
        c = c % ("client name ", *mode)
        logger.info(c)
        logger.info(b)
        for client_idx in range(cfg.num_clients):
            c = "|%25s|" % cfg.clients[client_idx].name
            for k in data_info_at_client[client_idx]:
                c += "%10s|" % (data_info_at_client[client_idx][k])
            logger.info(c)
        logger.info(b)

    @classmethod
    def log_client_cuda_info(cls, cfg, cuda_info_at_client):
        logger = cls.get_logger()
        for client_idx in range(cfg.num_clients):
            logger.info("Found %d device(s) at client %s" % (cuda_info_at_client[client_idx], 
                cfg.clients[client_idx].name))
            
    @classmethod
    def log_server_data_info(cls, data_info_at_server):
        mode = list(data_info_at_server.keys())
        logger = cls.get_logger()
        b = "|" + "-" * 10 + "|" + ("-" * 10 + "|") * len(mode)
        logger.info(b)
        c = "|%10s|" + "%10s|" * len(mode)
        c = c % (" ", *mode)
        logger.info(c)
        logger.info(b)
        c = "|%10s|" % "server"
        for k in data_info_at_server:
            c += "%10s|" % (data_info_at_server[k])
        logger.info(c)
        logger.info(b)

    @classmethod
    def save_checkpoint(cls, ckpt_name, state_dict):
        lgg = cls.__logger
        if type(ckpt_name) == int:
            file = os.path.join(lgg.dir, "checkpoint_%d.pt" % ckpt_name)
        else:
            file = os.path.join(lgg.dir, "%s.pt" % str(ckpt_name))
        torch.save(state_dict, file)


class ClientLogger:
    def __init__(self) -> None:
        self.info = {}
        self.events = {}
        self.timing = {}

    def to_dict(self):
        return {"events": self.events, "timing": self.timing, "info": self.info}

    def __get_step_str(self, step):
        return "epoch_%d" % (step + 1)

    def add_info(self, name: str, value, step=None):
        if step == None:
            self.info[name] = value
        else:
            step = self.__get_step_str(step)
            if step not in self.info:
                self.info[step] = {}
            self.info[step][name] = value

    def mark_event(self, name: str, step=None):
        time_stp = str(datetime.now())
        if step == None:
            self.events[name] = time_stp
        else:
            step = self.__get_step_str(step)
            if step not in self.events:
                self.events[step] = {}
            self.events[step][name] = time_stp

    def start_timer(self, name: str, step=None):
        if step == None:
            self.timing[name] = time.time()
        else:
            step = self.__get_step_str(step)
            if step not in self.timing:
                self.timing[step] = {}
            self.timing[step][name] = time.time()

    def stop_timer(self, name: str, step=None):
        if step == None:
            self.timing[name] = round(time.time() - self.timing[name], 3)
        else:
            step = self.__get_step_str(step)
            self.timing[step][name] = round(time.time() - self.timing[step][name], 3)

    @staticmethod
    def to_str(client_log):
        o = OmegaConf.create(client_log)
        return OmegaConf.to_yaml(o)


def get_eval_results_from_logs(logs):
    val_results = {}
    for client_idx in logs:
        val_results[client_idx] = {
            **logs[client_idx]['info']['train_info'],
            **logs[client_idx]["info"]["val_before_update_val_set"]
        }

    return val_results
