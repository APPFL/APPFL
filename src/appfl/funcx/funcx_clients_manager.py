import os.path as osp
import random
import time

from omegaconf import DictConfig, OmegaConf

from concurrent.futures import Future
from globus_compute_sdk import Executor
import concurrent

from collections import OrderedDict
from appfl.config import ClientTaskForLogging
from enum import Enum

from appfl.misc.logging import ClientLogger
from appfl.funcx.cloud_storage import CloudStorage, LargeObjectWrapper


class ClientStatus(Enum):
    UNAVAILABLE = 0
    AVAILABLE = 1
    RUNNING = 2


class ClientTask:
    def __init__(self, fut:Future, task_id:str, task_name: str, client_idx: int, start_time) -> None:
        self.fut = fut
        self.task_name = task_name
        self.client_idx = client_idx
        self.pendding = True
        self.success = False
        self.start_time = start_time
        self.end_time = -1
        self.log = {}
        self.task_id = task_id

class FuncXClient:
    def __init__(self, client_idx: int, client_cfg: OrderedDict):
        self.client_idx = client_idx
        self.client_cfg = client_cfg
        self.set_no_runing_task()

    @property
    def status(self):
        if self._status == ClientStatus.RUNNING:
            if self.future.done():
                self.set_no_runing_task()
        return self._status

    def set_no_runing_task(self):
        self._status = ClientStatus.AVAILABLE
        self.task_name = "N/A"
        self.executing_task_id = None
        self.future = None

    def submit_task(self, fx, exct_func, *args, callback=None, **kwargs):
        if self.status == ClientStatus.AVAILABLE:
            self._status = ClientStatus.RUNNING
            self.task_name = exct_func.__name__
            self.executing_task_id = str("%03d" % random.randint(1, 999))
            # def exct_func_wrapper(*args, **kwargs):
            #     return (exct_func(*args, **kwargs), self.executing_task_id)
            try:
                self.future = fx.submit(
                    exct_func,
                    *args,
                    **kwargs,
                    endpoint_id=self.client_cfg.endpoint_id,
                )
            except:
                pass
            if callback is not None:
                self.future.add_done_callback(callback)
            return self.executing_task_id
        else:
            return None


class APPFLFuncXTrainingClients:
    def __init__(self, cfg: DictConfig, fxc: FuncXClient, logger):
        self.cfg = cfg
        self.fxc = fxc
        self.executing_tasks = {}
        # Logging
        self.logger = logger
        self.clients = {
            client_idx: FuncXClient(client_idx, client_cfg)
            for client_idx, client_cfg in enumerate(self.cfg.clients)
        }
        # Config S3 bucket (if necessary)
        self.use_s3bucket = cfg.server.s3_bucket is not None
        if self.use_s3bucket:
            CloudStorage.init(
                cfg, temp_dir=osp.join(cfg.server.output_dir, "tmp"), logger=self.logger
            )
    
    def __register_task(self, fut, client_id, task_name):
        """Register new client task - call after task submission"""
        temp_task_id = "temp_task_id_%d" % len(self.executing_tasks)
        self.executing_tasks[temp_task_id] = ClientTask(
                fut = fut,
                task_name = task_name,
                task_id = temp_task_id,
                client_idx=client_id,
                start_time=time.time(),
            )
        
    def __update_task_id(self, old_id, new_id):
        if old_id == new_id:
            return
        self.executing_tasks[new_id] = self.executing_tasks[old_id]
        self.executing_tasks.pop(old_id)

    def __set_task_success_status(self, task_id, completion_time, client_log=None):
        self.executing_tasks[task_id].end_time = float(completion_time)
        self.executing_tasks[task_id].success = True
        self.executing_tasks[task_id].log = client_log

    def __finalize_task(self, task_id):
        # Save task to log file
        tsk = self.executing_tasks[task_id]
        self.cfg.logging_tasks.append(OmegaConf.structured(ClientTaskForLogging(
                task_id=tsk.task_id,
                task_name=tsk.task_name,
                client_idx=tsk.client_idx,
                start_time=tsk.start_time,
                end_time=tsk.end_time,
                log=tsk.log
            )))

        # Remove from executing tasks list
        self.executing_tasks.pop(task_id)

    def __process_client_results(self, results):
        """Process results from clients, download file from S3 if necessary"""
        if not CloudStorage.is_cloud_storage_object(results) or not self.use_s3bucket:
            return results
        else:
            return CloudStorage.download_object(
                results, to_device=self.cfg.server.device
            )

    def __handle_params(self, args, kwargs):
        """Parse function's parameters and upload to S3"""
        # Handling args
        _args = list(args)
        for i, obj in enumerate(_args):
            if type(obj) == LargeObjectWrapper:
                if not obj.can_send_directly:
                    _args[i] = CloudStorage.upload_object(obj)
                else:
                    _args[i] = obj.data
        args = tuple(_args)
        # Handling kwargs
        for k in kwargs:
            obj = kwargs[k]
            if type(obj) == LargeObjectWrapper:
                if not obj.can_send_directly:
                    kwargs[k] = CloudStorage.upload_object(obj)
                else:
                    kwargs[k] = obj.data
        return args, kwargs

    def __handle_funcx_result(self, res, task_id):
        """Handle returned results from Funcx"""
        client_log = {}
        if type(res) == tuple:
            res, client_log = res
            self.logger.info("--- Client logs ---\n" + ClientLogger.to_str(client_log))
        else:
            client_log = None

        res = self.__process_client_results(res)
        self.__set_task_success_status(task_id, time.time(), client_log)
        client_task = self.executing_tasks[task_id]
        self.logger.info(
            "Recieved results of task '%s' from %s."
            % (
                client_task.task_name,
                self.clients[client_task.client_idx].client_cfg.name,
            )
        )
        self.__finalize_task(task_id)
        return res, client_log

    def __handle_future_result(self, future):
        """Handle returned Future object from Funcx"""
        res = future.result()
        task_id = future.task_id
        return self.__handle_funcx_result(res, task_id)

    def __handle_dict_result(self, dict_result, task_id):
        """Handle returned dictionary object from Funcx"""
        return self.__handle_funcx_result(dict_result["result"], task_id)

    def send_task_to_all_clients(self, exct_func, *args, silent=False, **kwargs):
        """Broadcast an excutabl tasks with all agruments to all clients"""
        # Prepare args, kwargs before sending to clients
        args, kwargs = self.__handle_params(args, kwargs)
        # Register funcX function and create execution batch
        func_uuid = self.fxc.register_function(exct_func)
        batch = self.fxc.create_batch()
        
        for client_idx, client_cfg in enumerate(self.cfg.clients):
            # select device
            
            with Executor(endpoint_id=client_cfg.endpoint_id) as gce:
                fut = gce.submit(exct_func, self.cfg, client_idx, *args, **kwargs)
                self.__register_task(fut, client_idx, exct_func.__name__)

            # batch.add(
            #     func_uuid,
            #     client_cfg.endpoint_id,
            #     [self.cfg, client_idx, *args],
            #     kwargs,
            # )
        # Execute training tasks at clients
        # TODO: Assuming that all tasks do not have the same start time
        # task_ids = self.fxc.batch_run(batch)

        # Saving task ids
        # for i, task_id in enumerate(futs):
        #     self.__register_task(fut.task_id, i, exct_func.__name__)

        # Logging
        if not silent:
            for task_id in self.executing_tasks:
                self.logger.info(
                    "Task '%s' (id: %s) is assigned to %s."
                    % (
                        exct_func.__name__,
                        task_id,
                        self.cfg.clients[self.executing_tasks[task_id].client_idx].name,
                    )
                )
        return self.executing_tasks

    # def receive_sync_endpoints_updates(self):
    #     sync_results = {}
    #     for client_idx in self.clients:
    #         print(self.clients[client_idx].task_name)
    #         sync_results[client_idx] = self.__handle_funcx_future(self.clients[client_idx].future)
    #     return sync_results

    def receive_sync_endpoints_updates(self):
        client_results = OrderedDict()
        client_logs = OrderedDict()
        
        def _finalizing_task(task_id, fut):
            if fut.exception():
                # TODO: handling situations when training has errors
                client_results[task_id] = None
                self.logger.warning(
                    "Task %s on %s is failed with an error.",
                    task_id,
                    self.cfg.clients[self.executing_tasks[task_id].client_idx].name,
                )
                self.executing_tasks.pop(task_id)
                raise fut.result()
                 # Remove from executing tasks list
                
                # NOTE: I think you'll want to capture the exception, but I don't
                # think you want to raise *here*

                # Raise/Reraise the exception at client
                # excpt = results[task_id]["exception"]
                # if type(excpt) == Exception:
                #     raise excpt
                # else:
                #     results[task_id]["exception"].reraise()
            
            else:
                client_idx = self.executing_tasks[task_id].client_idx
                result, log = self.__handle_funcx_result(fut.result(), task_id)
                client_results[client_idx], client_logs[client_idx] = result, log     
                
        while self.executing_tasks:
            done_tasks = [c for c in self.executing_tasks.values() if c.fut.done()]
            pending_tasks = {
                c.fut: c.task_id
                for c in self.executing_tasks.values() if not c.fut.done()
            }

            for c in done_tasks:
                task_id = c.fut.task_id
                
                self.__update_task_id(c.task_id, task_id)
                c.task_id = task_id
                
                _finalizing_task(task_id, c.fut)

            for c in concurrent.futures.as_completed(list(pending_tasks.values())):
                c = pending_tasks.pop(c.fut)
                
                task_id = c.fut.task_id
                self.__update_task_id(c.task_id, task_id)
                c.task_id = task_id

                _finalizing_task(task_id, c.fut)
            
            
        # while not stop_aggregate:
        #     results = self.fxc.get_batch_result(list(self.executing_tasks))
        #     for task_id in results:
        #         if results[task_id]["pending"] == False:
        #             self.executing_tasks[task_id].pending = False
        #             if task_id in self.executing_tasks:
        #                 ## Training at client is succeeded
        #                 if "result" in results[task_id]:
        #                     client_idx = self.executing_tasks[task_id].client_idx
        #                     (
        #                         client_results[client_idx],
        #                         client_logs[client_idx],
        #                     ) = self.__handle_dict_result(results[task_id], task_id)
        #                 else:
        #                     # TODO: handling situations when training has errors
        #                     client_results[self.executing_tasks[task_id]] = None
        #                     self.logger.warning(
        #                         "Task %s on %s is failed with an error."
        #                         % (
        #                             task_id,
        #                             self.cfg.clients[
        #                                 self.executing_tasks[task_id].client_idx
        #                             ].name,
        #                         )
        #                     )
        #                     # Raise/Reraise the exception at client
        #                     excpt = results[task_id]["exception"]
        #                     if type(excpt) == Exception:
        #                         raise excpt
        #                     else:
        #                         results[task_id]["exception"].reraise()
        #     if len(self.executing_tasks) == 0:
        #         stop_aggregate = True
        
        # Managing a list of futures

        # while self.executing_tasks:
        #     done_futures = {
        #         c.task_id for 
        #     }
        return client_results, client_logs

    def register_async_call_back_func(
        self, call_back_func, invoke_async_func_on_complete=True
    ):
        # Callback function
        def __cbfn(res):
            task_id = res.task_id
            client_task = self.executing_tasks[task_id]
            # If the task is canceled
            if (
                res.cancel() == False or self.stopping_func() == False
            ):  # I changed from AND to OR
                self.logger.info(
                    "Recieved results of task '%s' from %s."
                    % (
                        client_task.task_name,
                        self.clients[client_task.client_idx].client_cfg.name,
                    )
                )
                # call the user's call back func
                call_back_func(res, client_task.client_idx)
                # TODO: get completion time stamp
                self.__set_task_success_status(task_id, time.time())
                if invoke_async_func_on_complete:
                    # Assign new task to client
                    self.run_async_task_on_client(client_task.client_idx)
            else:
                self.logger.info(
                    "Task '%s' from %s is canceled."
                    % (
                        client_task.task_name,
                        self.clients[client_task.client_idx].client_cfg.name,
                    )
                )
            self.__finalize_task(res.task_id)

        self.call_back_func = __cbfn

    def register_stopping_criteria(self, stopping_func):
        self.stopping_func = stopping_func

    def register_async_func(self, exct_func, *args, **kwargs):
        self.async_func = exct_func
        self.async_func_args = args
        self.async_func_kwargs = kwargs

    def run_async_task_on_client(self, client_idx):
        client = self.clients[client_idx]
        self.logger.info(
            "Async task '%s' is assigned to %s."
            % (self.async_func.__name__, client.client_cfg.name)
        )
        # Prepare args, kwargs before sending to clients
        args, kwargs = self.__handle_params(args, kwargs)
        # Send task to client
        task_id = client.submit_task(
            self.fx,
            self.async_func,
            self.cfg,
            client_idx,
            *self.async_func_args,
            **self.async_func_kwargs,
            callback=self.call_back_func if hasattr(self, "call_back_func") else None,
        )
        # Register new tasks
        self.__register_task(task_id, client_idx, self.async_func.__name__)

    def run_async_task_on_available_clients(self):
        # self.fx  = FuncXExecutor(self.fxc, batch_enabled = True)
        for client_idx in self.clients:
            if self.clients[client_idx].status == ClientStatus.AVAILABLE:
                self.run_async_task_on_client(client_idx)

    def shutdown_all_clients(self):
        self.logger.info("Shutting down all clients.")

        for client_idx in self.clients:
            self.clients[client_idx].future.cancel()

        self.fx.shutdown()
        self.logger.info("All clients have been shutted down successfully.")

        # Clean-up cloud storage
        CloudStorage.clean_up()