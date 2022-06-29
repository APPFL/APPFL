import abc
from omegaconf import DictConfig
from funcx import FuncXClient
import numpy as np
import torch.nn as nn
import copy
import time
import logging
from ..algorithm import *
from ..misc import *

from .funcx_client import client_training, client_validate_data
from .funcx_clients_manager import APPFLFuncXTrainingClients
from .helpers import appfl_funcx_save_log

class APPFLFuncXServer(abc.ABC):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        self.cfg = cfg
        self.fxc = fxc

        ## Logger for a server
        logger = logging.getLogger(__name__)
        self.logger = create_custom_logger(logger, cfg)

        ## assign number of clients
        self.cfg.num_clients = len(self.cfg.clients)

        ## funcX - APPFL training client
        self.trn_endps = APPFLFuncXTrainingClients(self.cfg, fxc, self.logger)
        
        ## Using tensorboard to visualize the test loss
        if cfg.use_tensorboard:
            from tensorboardX import SummaryWriter

            self.writer = SummaryWriter(
                comment=self.cfg.fed.args.optim + "_clients_nums_" + str(self.cfg.num_clients)
            )
        
        self.best_accuracy = 0.0

    def _validate_clients_data(self):
        """ Checking data at clients """
        ## Geting the total number of data samples at clients
        self.trn_endps.send_task_to_all_clients(client_validate_data)
        training_size_at_client = self.trn_endps.receive_sync_endpoints_updates()
        return training_size_at_client

    def _get_client_weights(self, training_size_at_client):
        total_num_data = 0
        for k in range(self.cfg.num_clients):
            total_num_data += training_size_at_client[k]
            #TODO: What if a client doesn't have any training samples
            self.logger.info("Client %s has %d training samples" % (self.cfg.clients[k].name, training_size_at_client[k])) 
        ## weight calculation
        weights = {}
        for k in range(self.cfg.num_clients):
            weights[k] = training_size_at_client[k] / total_num_data
        return weights

    def set_validation_dataset(self, test_data):
        """ Server test-set data loader"""
        if self.cfg.validation == True and len(test_data) > 0:
            self.test_dataloader = DataLoader(
                test_data,
                num_workers=self.cfg.num_workers,
                batch_size= self.cfg.test_data_batch_size,
                shuffle   = self.cfg.test_data_shuffle,
            )
        else:
            self.cfg.validation = False
    
    def _initialize_server_model(self):
        """ APPFL server """
        self.server  = eval(self.cfg.fed.servername)(
            self.weights, copy.deepcopy(self.model), self.loss_fn, self.cfg.num_clients, "cpu", **self.cfg.fed.args        
        )
        # Server model should stay on CPU for serialization
        self.server.model.to("cpu")
    
    def _initialize_training(self, model: nn.Module, loss_fn: nn.Module):
        self.model =  model
        self.loss_fn =loss_fn
    
    def _do_validation(self, step):
        """ Validation """
        validation_start = time.time()
        test_loss    = 0.0
        test_accuracy= 0.0
        if self.cfg.validation == True:
            # Move server model to GPU (if available) for validation inference 
            if self.cfg.server.device != "cpu":
                self.server.model.to(self.cfg.server.device)

            test_loss, test_accuracy = validation(self.server, self.test_dataloader)
            
            if self.cfg.server.device != "cpu":
                # Move back to cpu after validation
                self.server.model.to("cpu")

            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                self.writer.add_scalar("server_test_accuracy", test_accuracy, step)
                self.writer.add_scalar("server_test_loss", test_loss, step)
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy

        self.cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
        self.cfg["logginginfo"]["test_loss"]       = test_loss
        self.cfg["logginginfo"]["test_accuracy"]   = test_accuracy
        self.cfg["logginginfo"]["BestAccuracy"]    = self.best_accuracy

    def _finalize_training(self):
        appfl_funcx_save_log(self.cfg, self.logger)
        self.server.logging_summary(self.cfg, self.logger)

    def _save_checkpoint(self, step):
        """ Saving model"""
        if (step + 1) % self.cfg.checkpoints_interval == 0 or step + 1 == self.cfg.num_epochs:
            if self.cfg.save_model == True:
                save_model_iteration(t + 1, self.server.model, self.cfg)

    @abc.abstractmethod
    def _do_training(self):
        pass 

    @abc.abstractmethod
    def run(self, model: nn.Module, loss_fn: nn.Module):
        pass
    
class APPFLFuncXSyncServer(APPFLFuncXServer):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        super(APPFLFuncXSyncServer, self).__init__(cfg, fxc)
        cfg["logginginfo"]["comm_size"] = 1
    
    def run(self, model: nn.Module, loss_fn: nn.Module):
        # Set model, and loss function
        self._initialize_training(model, loss_fn)
        # Validate data at clients
        training_size_at_client = self._validate_clients_data()
        # Calculate weight
        self.weights = self._get_client_weights(training_size_at_client)
        # Initialze model at server
        self._initialize_server_model()
        # Do training
        self._do_training()
        # Wrap-up
        self._finalize_training()
    
    def _do_training(self):
        """ Looping over all epochs """
        start_time = time.time()
        for t in range(self.cfg.num_epochs):
            self.logger.info(" ====== Epoch [%d/%d] ====== " % (t+1, self.cfg.num_epochs))
            per_iter_start = time.time()
            """ Do one training steps"""
            """ Training """
            ## Get current global state
            global_state = self.server.model.state_dict()
            local_update_start = time.time()
            ## Boardcast global state and start training at funcX endpoints
            tasks   = self.trn_endps.send_task_to_all_clients(client_training,
                        self.weights, global_state, self.loss_fn)
        
            ## Aggregate local updates from clients
            local_states = []
            local_states.append(self.trn_endps.receive_sync_endpoints_updates())
            self.cfg["logginginfo"]["LocalUpdate_time"] = time.time() - local_update_start

            ## Perform global update
            global_update_start = time.time()
            self.server.update(local_states)
            self.cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start
            self.cfg["logginginfo"]["PerIter_time"]      = time.time() - per_iter_start
            self.cfg["logginginfo"]["Elapsed_time"]      = time.time() - start_time
            
            """ Validation """
            self._do_validation(t)

            self.server.logging_iteration(self.cfg, self.logger, t)

            """ Saving checkpoint """
            self._save_checkpoint(t)

class APPFLFuncXAsyncServer(APPFLFuncXServer):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        super(APPFLFuncXAsyncServer, self).__init__(cfg, fxc)
        cfg["logginginfo"]["comm_size"] = 1

    def _initialize_server_model(self):
        """ Initialize server with only 1 client """
        self.server  = eval(self.cfg.fed.servername)(
            self.weights, copy.deepcopy(self.model), self.loss_fn, 1, "cpu", **self.cfg.fed.args        
        )
        # Send server model to device
        self.server.model.to("cpu")

    def _get_client_weights(self):
        weights = {
            k: 1 / self.cfg.num_clients for k in range(self.cfg.num_clients)
        }
        return weights

    def run(self, model: nn.Module, loss_fn: nn.Module):
        # Set model, and loss function
        self._initialize_training(model, loss_fn)
        # Validate data at clients
        # training_size_at_client = self._validate_clients_data()
        # Calculate weight
        self.weights = self._get_client_weights()
        # Initialze model at server
        self._initialize_server_model()
        # Do training
        self._do_training()
        # Wrap-up
        # self._finalize_training()

    def _do_training(self):
        ## Get current global state
        global_state = self.server.model.state_dict()
        self.count_updates = 0
        stop_aggregate= False

        def global_update(res):
            self.logger.info("Async FL global model update at step %d " % self.count_updates)
            self.count_updates += 1
            # TODO: fix this
            client_results = {0: res.result()}
            local_states = [client_results]
            # Perform global update
            global_update_start = time.time()
            self.server.update(local_states)
            self.cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start

        self.trn_endps.register_async_call_back_funcn(global_update)
        self.trn_endps.register_async_func(
            client_training, 
            self.weights, global_state, self.loss_fn
        )
        # Assigning training tasks to all available clients
        self.trn_endps.run_async_task_on_available_clients()

        while (not stop_aggregate):
            if self.count_updates % 2 == 0:
                self._do_validation(self.count_updates)
            # Define some stopping criteria
            if self.count_updates >= 3:
                self.logger.info("Training is finished!")
                stop_aggregate = True
                