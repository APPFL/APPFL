import abc
from http import client
from appfl.funcx.cloud_storage import CloudStorage, LargeObjectWrapper
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

class APPFLFuncXServer(abc.ABC):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        self.cfg = cfg
        self.fxc = fxc

        ## Logger for a server
        self.logger      = mLogging.get_logger()
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

    def _validate_clients_data(self):
        """ Checking data at clients """
        ## Geting the total number of data samples at clients
        mode = ['train', 'val', 'test']
        self.trn_endps.send_task_to_all_clients(client_validate_data, mode)
        data_info_at_client = self.trn_endps.receive_sync_endpoints_updates()
        assert len(data_info_at_client) > 0, "Number of clients need to be larger than 0"
        ## Logging 
        mLogging.log_client_data_info(self.cfg, data_info_at_client)
        self.data_info_at_client = data_info_at_client

    def _set_client_weights(self, mode = "samples_size"):
        if  self.data_info_at_client is None:
            mode = "equal"
        if mode == "samples_size":
            assert self.data_info_at_client is not None, "Please call the validate clients' data first"
            total_num_data = 0
            for k in range(self.cfg.num_clients):
                total_num_data += self.data_info_at_client[k]['train']
            ## weight calculation
            weights = {}
            for k in range(self.cfg.num_clients):
                weights[k]      = self.data_info_at_client[k]['train'] / total_num_data
        elif mode == "equal":
            weights = {
                k: 1 / self.cfg.num_clients for k in range(self.cfg.num_clients)
            }
        else:
            raise NotImplementedError
        self.weights = weights

    def set_server_dataset(self, validation_dataset=None, testing_dataset=None):
        val_loader, test_loader = None, None
        val_size, test_size     = 0,0
        """ Server test-set data loader"""
        if self.cfg.server_do_validation: 
            val_loader = get_dataloader(self.cfg, validation_dataset, mode='val')
            val_size   = len(val_loader) if val_loader is not None else 0
        if self.cfg.server_do_testing:
            test_loader= get_dataloader(self.cfg, testing_dataset,    mode='test')
            test_size  = len(test_loader)if val_loader is not None else 0
        if val_loader is None:
            self.cfg.server_do_validation = False
            self.logger.warning("Validation dataset at server is empty")
        if test_loader is None:
            self.cfg.server_do_testing    = False
            self.logger.warning("Testing dataset at server is empty")

        mLogging.log_server_data_info({"val": val_size, "test": test_size})
        self.server_testing_dataloader    = test_loader
        self.server_validation_dataloader = val_loader

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
    
    def __evaluate_global_model_at_server(self, dataloader):
        return validation(self.server, dataloader)

    def __evaluate_global_model_at_clients(self, mode = 'val'):
        assert mode in ['val', 'test']
        global_state = self.server.model.state_dict()
        _  = self.trn_endps.send_task_to_all_clients(client_testing,
                            self.weights, LargeObjectWrapper(global_state, "server_state"), self.loss_fn)
        eval_results = self.trn_endps.receive_sync_endpoints_updates()
        # TODO: handle this, refactor evaluation code
        for client_idx in eval_results:
            eval_results[client_idx] = {
                'loss': eval_results[client_idx][0],
                'acc' : eval_results[client_idx][1]
            }
        return eval_results
        
    def _do_server_validation(self, step:int):
        """ Validation """
        validation_start = time.time()
        val_loss    = 0.0
        val_accuracy= 0.0
        if self.cfg.server_do_validation== True:
            # Move server model to GPU (if available) for validation inference 
            # TODO: change to val_dataloader
            val_loss, val_accuracy = self.__evaluate_global_model_at_server(self.server_validation_dataloader)
            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                self.writer.add_scalar("server_test_accuracy", val_accuracy, step)
                self.writer.add_scalar("server_test_loss", val_loss, step)
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy

        self.cfg["logginginfo"]["Validation_time"] = time.time() - validation_start
        self.cfg["logginginfo"]["test_loss"]       = val_loss
        self.cfg["logginginfo"]["test_accuracy"]   = val_accuracy
        self.cfg["logginginfo"]["BestAccuracy"]    = self.best_accuracy
        self.eval_logger.log_server_validation({'acc': val_accuracy, 'loss': val_loss}, step)

    def _do_server_testing(self):
        """Peform testing at server """
        if self.cfg.server_do_testing:
            test_loss, test_accuracy = self.__evaluate_global_model_at_server(self.server_testing_dataloader)
            self.eval_logger.log_server_testing({'acc': test_accuracy, 'loss': test_loss})

    def _do_client_validation(self, step:int):
        """Perform validation at clients"""
        if self.cfg.client_do_validation:
            validation_results = self.__evaluate_global_model_at_clients(mode='val')
            self.eval_logger.log_client_validation(validation_results, step)
    
    def _do_client_testing(self):
        """Perform tesint at clients """
        if self.cfg.client_do_testing:
            testing_results  = self.__evaluate_global_model_at_clients(mode='test')
            self.eval_logger.log_client_testing(testing_results)
    
    def _finalize_experiment(self):
        # save log file
        mLogging.save_funcx_log(self.cfg)
        self.server.logging_summary(self.cfg, self.logger)
        # clean-up cloud storage
        CloudStorage.clean_up()

    def _save_checkpoint(self, step):
        """ Saving model"""
        if (step + 1) % self.cfg.checkpoints_interval == 0 or step + 1 == self.cfg.num_epochs:
            if self.cfg.save_model == True:
                save_model_iteration(step + 1, self.server.model, self.cfg)

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
        # self._validate_clients_data()
        # Calculate weight
        self._set_client_weights()
        # Initialze model at server
        self._initialize_server_model()
        # Do training
        self._do_training()
        # Do client testing
        self._do_client_testing()
        # Do server testing
        self._do_server_testing()
        # Wrap-up
        self._finalize_experiment()
    
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
            _  = self.trn_endps.send_task_to_all_clients(client_training,
                        self.weights, LargeObjectWrapper(global_state, "server_state"), self.loss_fn)
        
            ## Aggregate local updates from clients
            local_states = [self.trn_endps.receive_sync_endpoints_updates()]
            self.cfg["logginginfo"]["LocalUpdate_time"] = time.time() - local_update_start

            ## Perform global update
            global_update_start = time.time()
            self.server.update(local_states)
            self.cfg["logginginfo"]["GlobalUpdate_time"] = time.time() - global_update_start
            self.cfg["logginginfo"]["PerIter_time"]      = time.time() - per_iter_start
            self.cfg["logginginfo"]["Elapsed_time"]      = time.time() - start_time
            
            """ Validation """
            if (t+1) % self.cfg.server_validation_step == 0:
                self._do_server_validation(t+1)
            
            if (t+1) % self.cfg.server_validation_step == 0:
                self._do_client_validation(t+1)

            self.server.logging_iteration(self.cfg, self.logger, t)
            """ Saving checkpoint """
            self._save_checkpoint(t)

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