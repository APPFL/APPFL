import abc
import copy
import time
import torch.nn as nn

from appfl.funcx.cloud_storage import LargeObjectWrapper
from funcx import FuncXClient

from appfl.misc import (
    DictConfig,
    mLogging,
    get_dataloader,
    validation,
    get_eval_results_from_logs,
)

from appfl.algorithm import *
from appfl.funcx.funcx_client import client_testing, client_validate_data
from appfl.funcx.funcx_clients_manager import APPFLFuncXTrainingClients


class APPFLFuncXServer(abc.ABC):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        self.cfg = cfg
        self.fxc = fxc

        ## Logger for a server
        self.logger = mLogging.get_logger()
        self.eval_logger = mLogging.get_eval_logger()

        ## assign number of clients
        self.cfg.num_clients = len(self.cfg.clients)

        ## funcX - APPFL training client
        self.trn_endps = APPFLFuncXTrainingClients(self.cfg, fxc, self.logger)

        ## Using tensorboard to visualize the test loss
        if cfg.use_tensorboard:
            self.writer = mLogging.get_tensorboard_writer()

        ## Runtime variables
        self.best_accuracy = 0.0
        self.data_info_at_client = None

        ## Save best checkpoint
        if self.cfg.save_best_checkpoint:
            if  self.cfg.higher_is_better:
                self.best_eval = -1e8  
                self.is_better = lambda x, y: x > y 
            else:
                self.best_eval = 1e8
                self.is_better = lambda x, y: x < y 

    def _run_sync_task(self, exc_func, *args, **kwargs):
        self.trn_endps.send_task_to_all_clients(exc_func, *args, **kwargs)
        return self.trn_endps.receive_sync_endpoints_updates()

    def _validate_clients_data(self):
        """Checking data at clients"""
        ## Geting the total number of data samples at clients
        mode = ["train", "val", "test"]
        data_info_at_client, _ = self._run_sync_task(client_validate_data, mode)
        assert (
            len(data_info_at_client) > 0
        ), "Number of clients need to be larger than 0"
        ## Logging
        mLogging.log_client_data_info(self.cfg, data_info_at_client)
        self.data_info_at_client = data_info_at_client

    def _set_client_weights(self, mode="samples_size"):
        assert (
            self.data_info_at_client is not None
        ), "Please call the validate clients' data first"
        if mode == "sample_size":
            total_num_data = 0
            for k in range(self.cfg.num_clients):
                total_num_data += self.data_info_at_client[k]["train"]
            ## weight calculation
            weights = {}
            for k in range(self.cfg.num_clients):
                weights[k] = self.data_info_at_client[k]["train"] / total_num_data
        elif mode == "equal":
            weights = {k: 1 / self.cfg.num_clients for k in range(self.cfg.num_clients)}
        else:
            raise NotImplementedError
        self.weights = weights

    def set_server_dataset(self, validation_dataset=None, testing_dataset=None):
        val_loader, test_loader = None, None
        val_size, test_size = 0, 0
        """ Server test-set data loader"""
        if self.cfg.server_do_validation:
            val_loader = get_dataloader(self.cfg, validation_dataset, mode="val")
            val_size = len(validation_dataset) if val_loader is not None else 0
        if self.cfg.server_do_testing:
            test_loader = get_dataloader(self.cfg, testing_dataset, mode="test")
            test_size = len(testing_dataset) if test_loader is not None else 0
        if val_loader is None:
            self.cfg.server_do_validation = False
            self.logger.warning("Validation dataset at server is empty")
        if test_loader is None:
            self.cfg.server_do_testing = False
            self.logger.warning("Testing dataset at server is empty")

        mLogging.log_server_data_info({"val": val_size, "test": test_size})
        self.server_testing_dataloader = test_loader
        self.server_validation_dataloader = val_loader

    def _initialize_server_model(self):
        """APPFL server"""
        self.server = eval(self.cfg.fed.servername)(
            self.weights,
            copy.deepcopy(self.model),
            self.loss_fn,
            self.cfg.num_clients,
            "cpu",
            **self.cfg.fed.args
        )
        # Server model should stay on CPU for serialization
        self.server.model.to("cpu")

    def _initialize_training(self, model: nn.Module, loss_fn: nn.Module):
        self.model = model
        self.loss_fn = loss_fn

    def __evaluate_global_model_at_server(self, dataloader):
        return validation(self.server, dataloader)

    def __evaluate_global_model_at_clients(self, mode="val"):
        assert mode in ["val", "test"]
        global_state = self.server.model.state_dict()
        eval_results, _ = self._run_sync_task(
            client_testing,
            self.weights,
            LargeObjectWrapper(global_state, "server_state"),
            self.loss_fn,
        )
        # TODO: handle this, refactor evaluation code
        
        for client_idx in eval_results:
            cli_eval = eval_results[client_idx][1] 
            cli_eval = cli_eval if type(cli_eval) == dict else {
                'test_acc': cli_eval
                }
            eval_results[client_idx] = {
                'test_loss': eval_results[client_idx][0],
                **cli_eval 
            }
        return eval_results

    def _do_server_validation(self, step: int):
        """Validation"""
        validation_start = time.time()
        val_loss = 0.0
        val_accuracy = 0.0
        if self.cfg.server_do_validation == True:
            # Move server model to GPU (if available) for validation inference
            # TODO: change to val_dataloader
            val_loss, val_accuracy = self.__evaluate_global_model_at_server(
                self.server_validation_dataloader
            )
            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                self.writer.add_scalar("server_test_accuracy", val_accuracy, step)
                self.writer.add_scalar("server_test_loss", val_loss, step)
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy

        self.cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
        self.cfg["logginginfo"]["test_loss"] = val_loss
        self.cfg["logginginfo"]["test_accuracy"] = val_accuracy
        self.cfg["logginginfo"]["BestAccuracy"] = self.best_accuracy
        self.eval_logger.log_server_validation(
            {"acc": val_accuracy, "loss": val_loss}, step
        )

    def _do_server_testing(self):
        """Peform testing at server"""
        if self.cfg.server_do_testing:
            test_loss, test_accuracy = self.__evaluate_global_model_at_server(
                self.server_testing_dataloader
            )
            self.eval_logger.log_server_testing(
                {"acc": test_accuracy, "loss": test_loss}
            )

    def _do_client_validation(self, step: int, client_logs):
        """Parse validation results at clients from client logs"""
        if self.cfg.client_do_validation:
            validation_results = get_eval_results_from_logs(client_logs)
            self.eval_logger.log_client_validation(validation_results, step)
            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                for client_idx in validation_results:
                    client_name = self.cfg.clients[client_idx].name
                    for val_k in validation_results[client_idx]:
                        self.writer.add_scalar(
                            "%s-%s" % (client_name, val_k),
                            validation_results[client_idx][val_k],
                            step,
                        )
            return validation_results

    def _do_client_testing(self):
        """Perform tesint at clients"""
        if self.cfg.client_do_testing:
            testing_results = self.__evaluate_global_model_at_clients(mode="test")
            self.eval_logger.log_client_testing(testing_results)

    def _finalize_experiment(self):
        mLogging.save_funcx_log(self.cfg)
        self.server.logging_summary(self.cfg, self.logger)

    def _save_checkpoint(self, step):
        """Saving model"""
        if (
            step + 1
        ) % self.cfg.checkpoints_interval == 0 or step + 1 == self.cfg.num_epochs:
            if self.cfg.save_model == True:
                mLogging.save_checkpoint(step + 1, self.server.model.state_dict())
                # save_model_iteration(step + 1, self.server.model.state_dict(), self.cfg)
    
    def _save_best_checkpoint(self, eval_dict):
        if self.cfg.save_best_checkpoint:
            # Temporally use the validation loss, it should be able to use any metric
            eval = 0.0
            for cli_idx in self.weights:
                eval += self.weights[cli_idx] * eval_dict[cli_idx]['val_loss']
            
            if eval < self.best_eval: 
                self.logger.info("Saving best checkpoint")
                self.best_eval = eval
                mLogging.save_checkpoint("best", self.server.model.state_dict())
    
    @abc.abstractmethod
    def _do_training(self):
        pass

    def _lr_step(self, step):
        if step == 0:
            return
        self.trn_endps.cfg.fed.args.optim_args.lr *= (
            self.cfg.fed.args.server_lr_decay_exp_gamma
        )
        self.logger.info(
            "Learing rate at step %d %.06f is set to "
            % (step + 1, self.trn_endps.cfg.fed.args.optim_args.lr)
        )

    def run(self, model: nn.Module, loss_fn: nn.Module, mode="train"):
        assert mode in ["train", "clients_testing"]
        # Set model, and loss function
        self._initialize_training(model, loss_fn)
        # Validate data at clients
        self._validate_clients_data()
        # Calculate weight
        self._set_client_weights(mode=self.cfg.fed.args.client_weights)
        # Initialze model at server
        self._initialize_server_model()
        if mode == "train":
            # Do training
            self._do_training()
        elif mode == "clients_testing":
            assert self.cfg.load_model == True
            self.cfg["logginginfo"]["GlobalUpdate_time"] = 0.0
            self.cfg["logginginfo"]["PerIter_time"] = 0.0
            self.cfg["logginginfo"]["Elapsed_time"] = 0.0
            self.cfg["logginginfo"]["Validation_time"] = 0.0
            self.cfg["logginginfo"]["test_loss"] = 0.0
            self.cfg["logginginfo"]["test_accuracy"] = 0.0
            self.cfg["logginginfo"]["BestAccuracy"] = 0.0
        # Do client testing
        self._do_client_testing()
        # Do server testing
        self._do_server_testing()
        # Wrap-up
        self._finalize_experiment()
