from omegaconf import DictConfig
from funcx import FuncXClient
import torch.nn as nn
from collections import OrderedDict
import logging
from appfl.misc import create_custom_logger

class APPFLFuncTrainingEndpoints:
    def __init__(self, cfg: DictConfig, fxc : FuncXClient, logger):
        self.cfg = cfg
        self.fxc = fxc
        self.executing_tasks = {}
        
        # Logging
        self.logger = logger
        
    def send_task_to_clients(self, exct_func, *args, silent = False, **kwargs):
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
        task_ids = self.fxc.batch_run(batch)
        
        ## Saving task ids 
        for i, task_id in enumerate(task_ids):
            self.executing_tasks[task_ids[i]] = i
        
        ## Logging
        if not silent:
            for task_id in  self.executing_tasks:
                self.logger.info("Task '%s' (id: %s) is assigned to %s." %(exct_func.__name__, task_id, self.cfg.clients[self.executing_tasks[task_id]].name))
        return self.executing_tasks

    def receive_sync_endpoints_updates(self):
        stop_aggregate    = False
        client_results    = OrderedDict()
        while (not stop_aggregate):
            results = self.fxc.get_batch_result(list(self.executing_tasks.keys()))
            for task_id in results:
                if results[task_id]['pending'] == False:
                    if task_id in self.executing_tasks:
                        ## Training at client is succeeded
                        if results[task_id]['status'] == "success": 
                            client_results[self.executing_tasks[task_id]] = results[task_id]['result']
                            
                            self.logger.info(
                            "Task %s on %s is completed successfully." % ( 
                                task_id, 
                                self.cfg.clients[self.executing_tasks[task_id]].name)
                            )
                        else:
                            # TODO: handling situations when training has errors
                            client_results[self.executing_tasks[task_id]] = None

                            self.logger.warning(
                            "Task %s on %s is failed with an error." % ( 
                                task_id, 
                                self.cfg.clients[self.executing_tasks[task_id]].name)
                            )
                    self.executing_tasks.pop(task_id)

            if len(self.executing_tasks) == 0:
                stop_aggregate = True
        return client_results