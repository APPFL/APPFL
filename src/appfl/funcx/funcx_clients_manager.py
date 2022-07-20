from omegaconf import DictConfig, OmegaConf
from funcx import FuncXClient, FuncXExecutor
import torch.nn as nn
from collections import OrderedDict
import time
from appfl.config import ClientTask
from enum import Enum
import asyncio
from appfl.funcx.cloud_storage import CloudStorage
import os.path as osp
import os
import torch
import pickle as pkl

class ClientStatus(Enum):
    UNAVAILABLE = 0
    AVAILABLE   = 1
    RUNNING     = 2
    # DONE        = 3

class FuncXClient:
    def __init__(self, client_idx:int, client_cfg: OrderedDict, status = ClientStatus.AVAILABLE):
        self._status: ClientStatus = ClientStatus.AVAILABLE
        self.client_idx = client_idx 
        self.client_cfg = client_cfg
        self.future   = None
        self.task_name= "N/A"

    @property
    def status(self):
        if (self._status == ClientStatus.RUNNING):
            if self.future.done():
                self._status = ClientStatus.AVAILABLE
        return self._status


    def submit_task(self, fx, exct_func, *args, callback = None, **kwargs ):
        if self.status == ClientStatus.AVAILABLE:
            self._status = ClientStatus.RUNNING
            self.task_name = exct_func.__name__
            try:
                self.future = fx.submit(
                    exct_func, *args, **kwargs,
                    endpoint_id = self.client_cfg.endpoint_id, 
                )
            except:
                pass
            self.future.add_done_callback(callback)
            return self.future.task_id
        else:
            return None
    
    # def cancel_task(self):
    #     return self.fx.cancel()
    # def get_result(self):
    #     if self.status   == ClientStatus.DONE:
    #         self._status  = ClientStatus.AVAILABLE
    #         self.task_name= "N/A"
    #         return self.future.result()  
    #     else:
    #         return None
    
class APPFLFuncXTrainingClients:
    def __init__(self, cfg: DictConfig, fxc : FuncXClient, logger):
        self.cfg = cfg
        self.fxc = fxc
        self.fx  = FuncXExecutor(fxc, batch_enabled = False)
        self.executing_tasks = {}
        
        # Logging
        self.logger  = logger
        self.clients = {
            client_idx: FuncXClient(client_idx, client_cfg)
            for client_idx, client_cfg in enumerate(self.cfg.clients)
        }

        # Config S3 bucket (if necessary)
        self.use_s3bucket = cfg.server.s3_bucket is not None
        
        if (self.use_s3bucket):
            CloudStorage.init(cfg.server)

    def __register_task(self, task_id, client_id, task_name):
        self.executing_tasks[task_id] =  OmegaConf.structured(ClientTask(
                    task_id    = task_id,
                    task_name  = task_name,
                    client_idx = client_id,
                    start_time = time.time()
                ))
    def __set_task_success_status(self, task_id, completion_time):
        self.executing_tasks[task_id].end_time= float(completion_time)
        self.executing_tasks[task_id].success = True

    def __finalize_task(self, task_id):
        # Save task to log file
        self.cfg.logging_tasks.append(self.executing_tasks[task_id])
        # Remove from executing tasks list
        self.executing_tasks.pop(task_id)
    
    def __process_client_results(self, results):
        if not CloudStorage.is_cloud_storage_object(results) or not self.use_s3bucket:
            return results
        else:
            _, object_name, file_name = CloudStorage.get_cloud_object_info(results)
            # Prepare cache dir
            cache_dir = osp.join(self.cfg.server.output_dir, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            file_path = osp.join(cache_dir, file_name)
            self.logger.info("Downloading object %s" % file_name)
            # Download file
            cs = CloudStorage.get_instance()
            cs.download(object_name,file_path)
            # Load files to memory
            file_ext = osp.splitext(osp.basename(file_name))[1]
            if  file_ext in ['.pt', '.pth']:
                results = torch.load(file_path)
            elif results in ['.pkl']:
                with open(file_path, "rb") as fi:
                    results = pkl.load(fi)
            # import ipdb; ipdb.set_trace()
            return results
        
    def send_task_to_all_clients(self, exct_func, *args, silent = False, **kwargs):
        ## Register funcX function and create execution batch 
        func_uuid = self.fxc.register_function(exct_func)
        batch     = self.fxc.create_batch()

        for client_idx, client_cfg in enumerate(self.cfg.clients):
            # select device
            batch.add(
                self.cfg,
                client_idx, # TODO: can work with other datasets
                *args,
                **kwargs,
                endpoint_id = client_cfg.endpoint_id, 
                function_id = func_uuid)
        
        ## Execute training tasks at clients
        #TODO: Assuming that all tasks do not have the same start time
        task_ids  = self.fxc.batch_run(batch)
        
        ## Saving task ids 
        for i, task_id in enumerate(task_ids):
            self.__register_task(
                task_id, i, exct_func.__name__
            )
            
        ## Logging
        if not silent:
            for task_id in  self.executing_tasks:
                self.logger.info("Task '%s' (id: %s) is assigned to %s." %(
                    exct_func.__name__, task_id, 
                    self.cfg.clients[self.executing_tasks[task_id].client_idx].name))
        return self.executing_tasks

    def receive_sync_endpoints_updates(self):
        stop_aggregate    = False
        client_results    = OrderedDict()
        while (not stop_aggregate):
            results = self.fxc.get_batch_result(list(self.executing_tasks))
            for task_id in results:
                if results[task_id]['pending'] == False:
                    self.executing_tasks[task_id].pending = False
                    if task_id in self.executing_tasks:
                        ## Training at client is succeeded
                        if results[task_id]['status'] == "success":
                            client_results[self.executing_tasks[task_id].client_idx] =self.__process_client_results(results[task_id]['result'])
                            
                            self.__set_task_success_status(task_id, results[task_id]["completion_t"])

                            # Logging                      
                            self.logger.info(
                            "Task %s on %s completed successfully." % ( 
                                task_id, 
                                self.cfg.clients[self.executing_tasks[task_id].client_idx].name)
                            )

                        else:
                            # TODO: handling situations when training has errors
                            client_results[self.executing_tasks[task_id]] = None
                            self.logger.warning(
                            "Task %s on %s is failed with an error." % ( 
                                task_id, 
                                self.cfg.clients[self.executing_tasks[task_id].client_idx].name)
                            )
                            # Raise/Reraise the exception at client
                            excpt = results[task_id]['exception']
                            if type(excpt) == Exception:
                                raise excpt
                            else:
                                results[task_id]['exception'].reraise()
                    
                    # Finalize task
                    self.__finalize_task(task_id)
            if len(self.executing_tasks) == 0:
                stop_aggregate = True
        return client_results
    
    def register_async_call_back_funcn(self, call_back_func):
        # Callback function
        def __cbfn(res):
            task_id = res.task_id
            client_task  = self.executing_tasks[task_id]
            # If the task is canceled
            if res.cancel() == False and self.stopping_func() == False:
                self.logger.info("Recieved results of task '%s' from %s." %(
                                    client_task.task_name, 
                                    self.clients[client_task.client_idx].client_cfg.name)
                    )
                # call the user's call back func
                call_back_func(res, client_task.client_idx)
                # TODO: get completion time stamp
                self.__set_task_success_status(task_id, time.time())
                # Assign new task to client
                self.run_async_task_on_client(client_task.client_idx)
            else:
                self.logger.info("Task '%s' from %s is canceled." % (
                    client_task.task_name, self.clients[client_task.client_idx].client_cfg.name)
                    )
            self.__finalize_task(res.task_id)
        self.call_back_func = __cbfn
    
    def register_stopping_criteria(self, stopping_func):
        self.stopping_func = stopping_func

    def register_async_func(self, exct_func, *args, **kwargs):
        self.async_func       = exct_func
        self.async_func_args  = args
        self.async_func_kwargs= kwargs

    def run_async_task_on_client(self, client_idx):
        client = self.clients[client_idx]
        self.logger.info("Async task '%s' is assigned to %s." %(
                        self.async_func.__name__, client.client_cfg.name)
                ) 
        # Send task to client
        task_id = client.submit_task(
            self.fx,
            self.async_func,
            self.cfg,
            client_idx,
            *self.async_func_args,
            **self.async_func_kwargs,
            callback = self.call_back_func
        ) 
        # Register new tasks
        self.__register_task(
            task_id, client_idx, self.async_func.__name__
        )
    
    def run_async_task_on_available_clients(self):
        for client_idx in self.clients:
            if self.clients[client_idx].status == ClientStatus.AVAILABLE:
                self.run_async_task_on_client(client_idx)
    
    def shutdown_all_clients(self):
        self.logger.info("Shutting down all clients.")
        
        for client_idx in self.clients:
            self.clients[client_idx].future.cancel()
        
        self.fx.shutdown()
        self.logger.info("All clients have been shutted down successfully.")
    
    # def run_loop(self):
    #     loop = asyncio.get_event_loop()
    #     try:
    #         loop.run_forever()
    #     except KeyboardInterrupt:
    #         pass
    #     finally:
    #         print("Ending loop")
    #         loop.close()
    
    # def get_async_result_from_clients(self):
    #     results = OrderedDict()
    #     for client_idx in self.clients:
    #         client = self.clients[client_idx]
    #         if client.status == ClientStatus.DONE:
    #             self.logger.info("Recieved results of task '%s' from %s." %(
    #                             client.task_name, client.client_cfg.name)
    #             )
    #             results[client_idx] = client.get_result()
    #     return results
                    