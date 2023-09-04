import time
import uuid
import os.path as osp
from collections import OrderedDict
from appfl.config import ClientTask
from appfl.misc.logging import ClientLogger
from omegaconf import DictConfig, OmegaConf
from globus_compute_sdk import Executor, Client
from .utils.s3_storage import CloudStorage, LargeObjectWrapper
from .utils.globus_compute_endpoint import GlobusComputeClientEndpoint, ClientEndpointStatus

class GlobusComputeCommunicator:
    def __init__(self, cfg: DictConfig, gcc : Client, logger):
        self.cfg = cfg
        self.gcc = gcc
        self.logger = logger
        self.executing_tasks = {}
        self.clients = {
            client_idx: GlobusComputeClientEndpoint(client_idx, client_cfg)
            for client_idx, client_cfg in enumerate(self.cfg.clients)
        }
        self.gcx = Executor(funcx_client=gcc, batch_enabled=True)
        self.use_s3bucket = cfg.server.s3_bucket is not None
        if self.use_s3bucket:
            self.logger.info('Using S3 bucket for model transfer.')
            CloudStorage.init(cfg, temp_dir= osp.join(cfg.server.output_dir, 'tmp'),logger= self.logger)

    def __register_task(self, task_id, client_id, task_name):
        """Register new client task to the list of executing tasks - call after task submission """
        self.executing_tasks[task_id] =  OmegaConf.structured(ClientTask(
            task_id    = task_id,
            task_name  = task_name,
            client_idx = client_id,
            start_time = time.time())
        )
        
    def __set_task_success_status(self, task_id, completion_time, client_log = None):
        """Change the status status of the given task to finished."""
        self.executing_tasks[task_id].end_time= float(completion_time)
        self.executing_tasks[task_id].success = True
        self.executing_tasks[task_id].log     = client_log

    def __finalize_task(self, task_id):
        """Finalize the given task."""
        # Save task to log file
        self.cfg.logging_tasks.append(self.executing_tasks[task_id])
        # Reset client
        self.clients[self.executing_tasks[task_id].client_idx].status
        # Remove from executing tasks list
        self.executing_tasks.pop(task_id)
        
    def __process_client_results(self, results):
        """Process results from clients, download file from S3 if necessary"""
        if not CloudStorage.is_cloud_storage_object(results) or not self.use_s3bucket:
            return results
        else:
            return CloudStorage.download_object(results, to_device=self.cfg.server.device, delete_cloud=True, delete_local=True)
    
    def __handle_params(self, args, kwargs):
        """Parse function's parameters and upload to S3 """
        # Handling args
        _args = list(args)
        for i, obj in enumerate(_args):
            if type(obj) == LargeObjectWrapper:
                if (not obj.can_send_directly) and self.use_s3bucket:
                    _args[i] = CloudStorage.upload_object(obj, register_for_clean=True)
                else:
                    _args[i] = obj.data
        args = tuple(_args)
        # Handling kwargs
        for k in kwargs:
            obj = kwargs[k]
            if type(obj) == LargeObjectWrapper:
                if (not obj.can_send_directly) and self.use_s3bucket:
                    kwargs[k] = CloudStorage.upload_object(obj, register_for_clean=True)
                else:
                    kwargs[k] = obj.data
        return args, kwargs

    def __handle_funcx_result(self, res, task_id, do_finalize=True):
        """Handle returned results from Funcx"""
        # Obtain the client logs
        client_log = {}
        if type(res) == tuple:
            res, client_log = res
            self.logger.info("--- Client logs ---\n" + ClientLogger.to_str(client_log))
        else:
            client_log = None
        # Download client results from S3 bucker if necessary
        res = self.__process_client_results(res)
        # Change the status of the finished task
        self.__set_task_success_status(task_id, time.time(), client_log)
        # Log the receipt of the task results
        client_name = self.clients[self.executing_tasks[task_id].client_idx].client_cfg.name
        client_task_name = self.executing_tasks[task_id].task_name
        self.logger.info("Recieved results of task '%s' from %s." %(client_task_name, client_name))
        # Finalize the experiment if necessary
        if do_finalize:
            self.__finalize_task(task_id)
        return res, client_log

    def __handle_future_result(self, future):
        """Handle returned Future object from Funcx"""
        res = future.result()
        task_id = future.task_id
        return self.__handle_funcx_result(res, task_id, do_finalize=False)
    
    def __handle_dict_result(self, dict_result, task_id):
        """Handle returned dictionary object from Funcx"""
        return self.__handle_funcx_result(dict_result['result'], task_id)

    def send_task_to_all_clients(self, exct_func, *args, silent = False, **kwargs):
        """Broadcast an excutable task with all agruments to all federated learning clients"""
        # Prepare args, kwargs before sending to clients
        args, kwargs = self.__handle_params(args, kwargs)
        # Register funcX function and create an execution batch
        func_uuid = self.gcc.register_function(exct_func)
        batch     = self.gcc.create_batch()
        # Execute training tasks at clients
        for client_idx, client_cfg in enumerate(self.cfg.clients):
            if self.use_s3bucket and exct_func.__name__ == 'client_training':
                local_model_key = str(uuid.uuid4()) + f"_client_state_{client_idx}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                kwargs['local_model_key'] = local_model_key
                kwargs['local_model_url'] = local_model_url
            batch.add(
                func_uuid,
                client_cfg.endpoint_id,
                args=(self.cfg, client_idx, *args),
                kwargs=kwargs)
        task_ids  = self.gcc.batch_run(batch)
        # Saving task ids 
        for i, task_id in enumerate(task_ids):
            self.__register_task(task_id, i, exct_func.__name__)
        # Logging
        if not silent:
            for task_id in  self.executing_tasks:
                self.logger.info("Task '%s' (id: %s) is assigned to %s." %(
                    exct_func.__name__, task_id, 
                    self.cfg.clients[self.executing_tasks[task_id].client_idx].name))
        return self.executing_tasks
    
    def receive_sync_endpoints_updates(self):
        """Receive synchronous updates from all client endpoints."""
        client_results = OrderedDict()
        client_logs    = OrderedDict()
        while True:
            results = self.gcc.get_batch_result(list(self.executing_tasks))
            if len(results) != len(list(self.executing_tasks)):
                raise Exception("Exception occurs on client side, stop the training!")
            for task_id in results:
                if results[task_id]['pending'] == False:
                    if task_id in self.executing_tasks:
                        self.executing_tasks[task_id].pending = False
                        # Training at client is succeeded
                        if 'result' in results[task_id]:
                            client_idx = self.executing_tasks[task_id].client_idx
                            client_results[client_idx], client_logs[client_idx] = self.__handle_dict_result(results[task_id], task_id)
                        else:
                            # TODO: handling situations when training has errors
                            client_results[self.executing_tasks[task_id]] = None
                            self.logger.warning(
                            "Task %s on %s is failed with an error." % ( 
                                task_id, self.cfg.clients[self.executing_tasks[task_id].client_idx].name))
                            # Raise/Reraise the exception at client
                            excpt = results[task_id]['exception']
                            if type(excpt) == Exception:
                                raise excpt
                            else:
                                results[task_id]['exception'].reraise()
            if len(self.executing_tasks) == 0: break
        return client_results, client_logs
    
    def register_async_call_back_func(self, call_back_func, invoke_async_func_on_complete = True):
        """
        Register callback function for asynchronous tasks sent to federated learning clients.
        Args:
            call_back_func (function): callback function invoked after the sent task finishes.
            invoke_async_func_on_complete (bool): whether to invoke a new asynchrnous task to clients after completion of previous task.
        """
        def __cbfn(res):
            # Return directly if the task is already canceled
            if res.task_id is None or res.task_id not in self.executing_tasks: 
                return
            task_id = res.task_id
            client_task  = self.executing_tasks[task_id]
            task_name = client_task.task_name
            client_name = self.clients[client_task.client_idx].client_cfg.name
            if self.stopping_func() == False:
                result, client_log = self.__handle_future_result(res)
                # Call the provided callback function to do global model update
                call_back_func(result, client_task.client_idx, client_log)
                # Update the task status
                self.__set_task_success_status(task_id, time.time(), client_log)
                # Assign new task to client
                if invoke_async_func_on_complete:
                    self.run_async_task_on_client(client_task.client_idx)
                # Do the server validation after assigning new tasks to the client to save time
                self.validation_func()
            else:
                if res.cancelled:
                    self.logger.info("Task '%s' (id: %s) from %s is canceled." % (task_name, task_id, client_name))
                else:
                    res.cancel()
                    self.logger.info("Task '%s' (id: %s) from %s is being canceled." % (task_name, task_id, client_name))

            self.__finalize_task(res.task_id)
        self.call_back_func = __cbfn
    
    def register_stopping_criteria(self, stopping_func):
        self.stopping_func = stopping_func

    def register_validation_func(self, validation_func):
        self.validation_func = validation_func

    def register_async_func(self, exct_func, *args, **kwargs):
        self.async_func        = exct_func
        self.async_func_args   = args
        self.async_func_kwargs = kwargs

    def run_async_task_on_client(self, client_idx):
        """Run asynchronous task on a certain clients."""
        client = self.clients[client_idx]
        self.logger.info("Async task '%s' is assigned to %s." %(self.async_func.__name__, client.client_cfg.name)) 
        self.async_func_args, self.async_func_kwargs = self.__handle_params(self.async_func_args, self.async_func_kwargs)
        if self.use_s3bucket and self.async_func.__name__ == 'client_training':
            local_model_key = str(uuid.uuid4()) + f"_client_state_{client_idx}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            self.async_func_kwargs['local_model_key'] = local_model_key
            self.async_func_kwargs['local_model_url'] = local_model_url
        # Send task to client
        task_id = client.submit_task(
            self.gcx,
            self.async_func,
            self.cfg,
            client_idx,
            *self.async_func_args,
            **self.async_func_kwargs,
            callback = self.call_back_func if hasattr(self, 'call_back_func') else None
        ) 
        # Register new tasks
        self.__register_task(task_id, client_idx, self.async_func.__name__)
    
    def run_async_task_on_available_clients(self):
        """Run asynchronous task on all available clients."""
        for client_idx in self.clients:
            if self.clients[client_idx].status == ClientEndpointStatus.AVAILABLE:
                self.run_async_task_on_client(client_idx)
    
    def shutdown_all_clients(self):
        """Cancel all the running tasks on the clients and shutdown the funcx executor."""
        self.logger.info("Shutting down all clients.")
        for client_idx in self.clients:
            if self.clients[client_idx].future is not None:
                self.clients[client_idx].future.cancel()
                # print(f"Cancelling the future {self.clients[client_idx].future}")
                # try:
                #     self.clients[client_idx].future.cancel()
                #     print(f"Finish cancelling the furture: {self.clients[client_idx].future}")
                # except:
                #     print(f"{self.clients[client_idx].future} is already canceled!")
        self.gcx.shutdown()
        self.logger.info("All clients have been shutted down successfully.")
        # Clean-up cloud storage
        CloudStorage.clean_up()
