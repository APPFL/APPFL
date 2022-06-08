from omegaconf import DictConfig
from funcx import FuncXClient
import torch.nn as nn

class APPFLFuncXServer:
    def __init__(self, cfg: DictConfig, fxc : FuncXClient):
        self.cfg = cfg
        self.fxc = fxc
        self.executing_tasks = {}

    def send_task_to_clients(self, exct_func, *args, **kwargs):
        ## Register funcX function and create execution batch 
        func_uuid = self.fxc.register_function(exct_func)
        batch     = self.fxc.create_batch()
        for client_cfg in self.cfg.clients:
            # import ipdb; ipdb.set_trace()
            batch.add(
                self.cfg,
                client_cfg.data_folder, # TODO: can work with other datasets
                *args,
                endpoint_id = client_cfg.endpoint_id, 
                function_id = func_uuid)
        
        ## Execute training tasks at clients
        task_ids = self.fxc.batch_run(batch)
        
        ## Saving task ids 
        for i, task_id in enumerate(task_ids):
            self.executing_tasks[task_ids[i]] = self.cfg.clients[i].name
        
        ## Logging
        for task_id in  self.executing_tasks:
            print("Task id %s is assigned to %s." %(task_id, self.executing_tasks[task_id]))
        return self.executing_tasks

    def receive_sync_client_updates(self):
        stop_aggregate    = False
        client_results    = []
        while (not stop_aggregate):
            results = self.fxc.get_batch_result(list(self.executing_tasks.keys()))
            for task_id in results:
                if results[task_id]['pending'] == False:
                    if task_id in self.executing_tasks:
                        print("Training task %s on %s is completed with status '%s'." % ( 
                            task_id, 
                            self.executing_tasks[task_id],
                            results[task_id]['status']))
                        
                        if results[task_id]['status'] != "failed": 
                            client_results.append(results[task_id]['result'])
                    self.executing_tasks.pop(task_id)

            if len(self.executing_tasks) == 0:
                stop_aggregate = True
        return client_results