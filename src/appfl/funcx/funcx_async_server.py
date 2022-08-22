from appfl.funcx.cloud_storage import LargeObjectWrapper
from omegaconf import DictConfig
from funcx import FuncXClient
import numpy as np
import torch.nn as nn
import copy
import time
from ..algorithm import *
from ..misc import *

from .funcx_client import client_training, client_testing, client_validate_data
from .funcx_clients_manager import APPFLFuncXTrainingClients
from .funcx_server import APPFLFuncXServer


class APPFLFuncXAsyncServer(APPFLFuncXServer):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        super(APPFLFuncXAsyncServer, self).__init__(cfg, fxc)
        cfg["logginginfo"]["comm_size"] = 1

        # Save the version of global model initialized at each client 
        self.client_init_step ={
            i : 0 for i in range(self.cfg.num_clients)
        }

    def _initialize_server_model(self):
        """ Initialize server with only 1 client """
        self.server  = eval(self.cfg.fed.servername)(
            copy.deepcopy(self.model), self.loss_fn, 1, "cpu", **self.cfg.fed.args, weights = self.weights      
        )
        # Send server model to device
        self.server.model.to("cpu")

    def run(self, model: nn.Module, loss_fn: nn.Module):
        # TODO: combine into one run function
        # Set model, and loss function
        self._initialize_training(model, loss_fn)
        # Validate data at clients
        # training_size_at_client = self._validate_clients_data()
        # Calculate weight
        self.weights = None
        # Initialze model at server
        self._initialize_server_model()
        # Do training
        self._do_training()
        # Do client testing
        self._do_client_testing()
        # Wrap-up
        self._finalize_experiment()
        # Shutdown all clients
        self.trn_endps.shutdown_all_clients()

    def _do_training(self):
        ## Get current global state
        stop_aggregate     = False
        start_time         = time.time()
        def global_update(res, client_idx):
            # TODO: fix this
            client_results = {0: res}
            local_states = [client_results]
            # TODO: fix local update time
            self.cfg["logginginfo"]["LocalUpdate_time"]  = 0
            self.cfg["logginginfo"]["PerIter_time"]      = 0
            self.cfg["logginginfo"]["Elapsed_time"]      = 0
            
            # TODO: add semaphore to protect this update operator
            # Perform global update
            global_update_start = time.time()
            init_step           = self.client_init_step[client_idx]
            
            self.server.update(local_states, init_step = init_step)
            global_step         = self.server.global_step
            
            self.cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start
            self.logger.info("Async FL global model updated. GLB step = %02d | Staleness = %02d" 
                                % (global_step, global_step - init_step - 1))

            # Save new init step of client
            self.client_init_step[client_idx] = self.server.global_step

            # Update callback func
            self.trn_endps.register_async_func(
                client_training, 
                self.weights, self.server.model.state_dict(), self.loss_fn
            )

            # Training eval log
            self.server.logging_iteration(self.cfg, self.logger, global_step - 1)
            
            # Saving checkpoint
            self._save_checkpoint(global_step -1)

        def stopping_criteria():
            return self.server.global_step > self.cfg.num_epochs
        
        # Register callback function: global_update
        self.trn_endps.register_async_call_back_func(global_update)
        # Register async function: client training
        self.trn_endps.register_async_func(
            client_training, 
            self.weights, self.server.model.state_dict(), self.loss_fn
        )
        # Register the stopping criteria
        self.trn_endps.register_stopping_criteria(
            stopping_criteria
        )
        # Start asynchronous FL
        start_time = time.time()
        # Assigning training tasks to all available clients
        self.trn_endps.run_async_task_on_available_clients()

        # Run async event loop
        while (not stop_aggregate):
            if self.server.global_step % 2 == 0:
                self._do_server_validation(self.server.global_step)
            
            # Define some stopping criteria
            if stopping_criteria():
                self.logger.info("Training is finished!")
                stop_aggregate = True

        self.cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time