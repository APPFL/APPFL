import abc
import copy
import time
import uuid
import traceback
import torch.nn as nn
from .misc import *
from .algorithm import *
from globus_compute_sdk import Client
from omegaconf import DictConfig
from .comm.globus_compute import GlobusComputeCommunicator
from .comm.globus_compute.utils.s3_storage import LargeObjectWrapper
from .comm.globus_compute.globus_compute_client_function import client_validate_data, client_testing, client_training

class APPFLGlobusComputeServer(abc.ABC):
    def __init__(self, cfg: DictConfig, gcc: Client):
        self.cfg = cfg
        self.gcc = gcc
        self.cfg.num_clients = len(self.cfg.clients)

        # Logger for a server
        self.logger      = mLogging.get_logger()
        self.eval_logger = mLogging.get_eval_logger()

        # Globus Compute communicator
        self.communicator = GlobusComputeCommunicator(self.cfg, gcc, self.logger)
        
        # Using tensorboard to visualize the test loss
        if cfg.use_tensorboard:
            self.writer = mLogging.get_tensorboard_writer()

        # Runtime variables
        self.best_accuracy = 0.0
        self.data_info_at_client = None
    
    def _initialize_training(self, model: nn.Module, loss_fn: nn.Module):
        self.model =  model
        self.loss_fn =loss_fn
    
    def _validate_clients_data(self):
        mode = ['train', 'val', 'test']
        self.communicator.send_task_to_all_clients(client_validate_data, mode)
        data_info_at_client, _ = self.communicator.receive_sync_endpoints_updates()
        assert len(data_info_at_client) > 0, "Number of clients need to be larger than 0"
        mLogging.log_client_data_info(self.cfg, data_info_at_client)
        self.data_info_at_client = data_info_at_client

    def _set_client_weights(self, mode = "samples_size"):
        assert self.data_info_at_client is not None, "Please call the validate clients' data first"
        if mode == "sample_size":
            total_num_data = 0
            for k in range(self.cfg.num_clients):
                total_num_data += self.data_info_at_client[k]['train']
            weights = {}
            for k in range(self.cfg.num_clients):
                weights[k] = self.data_info_at_client[k]['train'] / total_num_data
        elif mode == "equal":
            weights = {k: 1 / self.cfg.num_clients for k in range(self.cfg.num_clients)}
        else:
            raise NotImplementedError
        self.weights = weights

    def _initialize_server_model(self):
        self.server  = eval(self.cfg.fed.servername)(
            self.weights, 
            copy.deepcopy(self.model), 
            self.loss_fn, 
            self.cfg.num_clients, 
            "cpu", 
            **self.cfg.fed.args        
        )
        # Server model should stay on CPU for serialization
        self.server.model.to("cpu")
    
    @abc.abstractmethod
    def _do_training(self):
        pass 

    def _do_client_testing(self):
        """Perform tesint at clients """
        # TODO: fix bug here
        return
        if self.cfg.client_do_testing:
            testing_results  = self.__evaluate_global_model_at_clients(mode='test')
            self.eval_logger.log_client_testing(testing_results)
    
    def _do_server_testing(self):
        """Peform testing at server """
        if self.cfg.server_do_testing:
            test_loss, test_accuracy = validation(self.server, self.server_testing_dataloader)
            self.eval_logger.log_server_testing({'acc': test_accuracy, 'loss': test_loss})

    def _finalize_experiment(self):
        mLogging.save_funcx_log(self.cfg)
        self.server.logging_summary(self.cfg, self.logger)

    def set_server_dataset(self, validation_dataset=None, testing_dataset=None):
        val_loader, test_loader = None, None
        val_size, test_size     = 0,0
        if self.cfg.server_do_validation: 
            val_loader = get_dataloader(self.cfg, validation_dataset, mode='val')
            val_size   = len(validation_dataset) if val_loader is not None else 0
        if self.cfg.server_do_testing:
            test_loader= get_dataloader(self.cfg, testing_dataset,    mode='test')
            test_size  = len(testing_dataset)if test_loader is not None else 0
        if val_loader is None:
            self.cfg.server_do_validation = False
            self.logger.warning("Validation dataset at server is empty")
        if test_loader is None:
            self.cfg.server_do_testing    = False
            self.logger.warning("Testing dataset at server is empty")

        mLogging.log_server_data_info({"val": val_size, "test": test_size})
        self.server_testing_dataloader    = test_loader
        self.server_validation_dataloader = val_loader

    def __evaluate_global_model_at_clients(self, mode = 'val'):
        assert mode in ['val', 'test']
        global_state = self.server.model.state_dict()
        self.communicator.send_task_to_all_clients(
            client_testing,
            self.weights,
            LargeObjectWrapper(global_state, "server_state"),
            self.loss_fn
        )
        eval_results, _ = self.communicator.receive_sync_endpoints_updates()        
        for client_idx in eval_results:
            cli_eval = eval_results[client_idx][1] 
            cli_eval = cli_eval if type(cli_eval) == dict else {'test_acc': cli_eval}
            eval_results[client_idx] = {'test_loss': eval_results[client_idx][0], **cli_eval}
        return eval_results
        
    def _do_server_validation(self, step:int):
        validation_start = time.time()
        val_loss    = 0.0
        val_accuracy= 0.0
        if self.cfg.server_do_validation== True:
            # Move server model to GPU (if available) for validation inference 
            # TODO: change to val_dataloader
            val_loss, val_accuracy = validation(self.server, self.server_validation_dataloader)
            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                self.writer.add_scalar("server_test_accuracy", val_accuracy, step)
                self.writer.add_scalar("server_test_loss", val_loss, step)
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy

        self.cfg.logginginfo.Validation_time = time.time() - validation_start
        self.cfg.logginginfo.test_loss = val_loss
        self.cfg.logginginfo.test_accuracy = val_accuracy
        self.cfg.logginginfo.BestAccuracy = self.best_accuracy
        self.eval_logger.log_server_validation({'acc': val_accuracy, 'loss': val_loss}, step)

    def _parse_client_logs(self, step:int, client_logs):
        """Parse validation results at clients from client logs"""
        if self.cfg.client_do_validation:
            validation_results = get_eval_results_from_logs(client_logs)
            self.eval_logger.log_client_validation(validation_results, step)
            if self.cfg.use_tensorboard:
                # Add them to tensorboard
                for client_idx in validation_results:
                    client_name = self.cfg.clients[client_idx].name
                    for val_k in validation_results[client_idx]:
                        self.writer.add_scalar("%s-%s" % (client_name, val_k), validation_results[client_idx][val_k], step)

    def _save_checkpoint(self, step):
        """ Saving model"""
        if (step + 1) % self.cfg.checkpoints_interval == 0 or step + 1 == self.cfg.num_epochs:
            if self.cfg.save_model == True:
                mLogging.save_checkpoint(step + 1, self.server.model.state_dict())
                # save_model_iteration(step + 1, self.server.model.state_dict(), self.cfg)

    def _lr_step(self, step):
        # TODO: This function is too low level, the communicator should provide interface
        if step == 0:
            return
        self.communicator.cfg.fed.args.optim_args.lr *=  self.cfg.fed.args.server_lr_decay_exp_gamma
        self.logger.info("Learing rate at step %d is set to %.06f" % (step + 1, self.communicator.cfg.fed.args.optim_args.lr))

    def run(self, model: nn.Module, loss_fn: nn.Module, mode='train'):
        assert mode in ['train', 'clients_testing']
        # TODO: Also need to set the validation metric
        self._initialize_training(model, loss_fn)
        self._validate_clients_data()
        self._set_client_weights(mode=self.cfg.fed.args.client_weights)
        self._initialize_server_model()
        if mode == "train":
            self._do_training()
        elif mode == 'clients_testing':
            assert self.cfg.load_model == True
            self.cfg.logginginfo.GlobalUpdate_time = 0.0
            self.cfg.logginginfo.PerIter_time = 0.0
            self.cfg.logginginfo.Elapsed_time = 0.0
            self.cfg.logginginfo.Validation_time = 0.0
            self.cfg.logginginfo.test_loss = 0.0
            self.cfg.logginginfo.test_accuracy = 0.0
            self.cfg.logginginfo.BestAccuracy = 0.0
        self._do_client_testing()
        self._do_server_testing()
        self._finalize_experiment()
        self.communicator.shutdown_all_clients()

    def cleanup(self):
        self.communicator.shutdown_all_clients()

class APPFLGlobusComputeSyncServer(APPFLGlobusComputeServer):
    def __init__(self, cfg: DictConfig, gcc: Client):
        cfg.logginginfo.comm_size = 1       # TODO: comm_size is not making sense here
        super(APPFLGlobusComputeSyncServer, self).__init__(cfg, gcc)
    
    def _do_training(self):
        start_time = time.time()
        server_model_basename = str(uuid.uuid4()) + "_server_state"
        for t in range(self.cfg.num_epochs):
            self.logger.info(" ====== Epoch [%d/%d] ====== " % (t+1, self.cfg.num_epochs))
            per_iter_start = time.time()
            global_state = self.server.model.state_dict()
            self._lr_step(t)
            self.communicator.send_task_to_all_clients(
                client_training,
                self.weights,
                LargeObjectWrapper(global_state, f"{server_model_basename}_{t}"),
                self.loss_fn,
                do_validation = self.cfg.client_do_validation
            )
            local_states, client_logs = self.communicator.receive_sync_endpoints_updates()
            local_states = [local_states]
            self._parse_client_logs(t, client_logs)
            self.cfg.logginginfo.LocalUpdate_time = time.time() - per_iter_start

            # Perform global update
            global_update_start = time.time()
            self.server.update(local_states)
            self.cfg.logginginfo.GlobalUpdate_time = time.time() - global_update_start
            self.cfg.logginginfo.PerIter_tim = time.time() - per_iter_start
            self.cfg.logginginfo.Elapsed_time = time.time() - start_time
            
            if (t+1) % self.cfg.server_validation_step == 0:
                self._do_server_validation(t+1)

            self.server.logging_iteration(self.cfg, self.logger, t)
            self._save_checkpoint(t)

def run_server(
    cfg: DictConfig, 
    model: nn.Module,
    loss_fn: nn.Module,
    gcc: Client,
    test_data: Dataset = Dataset(),
    val_data : Dataset = Dataset(),
    mode = 'train'
    ):
    serv = APPFLGlobusComputeSyncServer(cfg, gcc)
    try:
        serv.set_server_dataset(validation_dataset=val_data, testing_dataset= test_data) 
        serv.run(model, loss_fn, mode)
    except Exception as e:
        traceback.print_exc()
        print("Training fails, cleaning things up... ...")
        serv.cleanup()