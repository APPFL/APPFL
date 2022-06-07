import numpy as np
import torch.nn as nn
from .misc import *
from .algorithm import *
import time

from funcx import FuncXClient, FuncXExecutor

def rand_int():
    import random
    import numpy
    return random.randint(0,10)
    # import numpy as np
    # return np.random.randint(10)

def run_client(
    # cfg: DictConfig,
    # model: nn.Module,
    # loss_fn: nn.Module,
    fxc : FuncXClient,
    endpoint: str
    # num_clients: int,
    # train_data: Dataset,
    # test_data: Dataset = Dataset(),
):
    func_uuid = fxc.register_function(rand_int)
    batch     = fxc.create_batch()
    for i in range(10):
        batch.add(endpoint_id = endpoint, function_id = func_uuid)
    
    # Start training at clients
    batch_res = fxc.batch_run(batch) 
    stop_aggregate = False
    finished_tasks = set()
    client_results = []
    # Waiting for results from clients
    while (not stop_aggregate):
        results = fxc.get_batch_result(batch_res)
        for task_id in results:
            if results[task_id]['pending'] == False:
                if task_id not in finished_tasks:
                    finished_tasks.add(task_id)
                    print("[%02d/%02d] Training task %s is completed with status %s" % (
                        len(finished_tasks), len(results), 
                        task_id, 
                        results[task_id]['status']))
                    if results[task_id]['status'] != "failed": 
                        client_results.append(results[task_id]['result'])
        time.sleep(1)
        if len(finished_tasks) == len(results):
            stop_aggregate = True
    print("Final results: ", client_results)