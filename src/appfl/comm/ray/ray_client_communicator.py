import time

import ray
from omegaconf import OmegaConf
from appfl.comm.utils.executor import (
    get_sample_size_executor,
    data_readiness_report_executor,
    train_executor,
)
from appfl.comm.utils.config import ClientTask


@ray.remote
class RayClientCommunicator:
    def __init__(self, server_agent_config, client_agent_config):
        self.client_config = OmegaConf.merge(
            server_agent_config.client_configs, client_agent_config
        )


    def spinup(self, task: ClientTask):
        task.task_execution_start_time = time.time()
        task.result = "Launched"
        task.task_execution_finish_time = time.time()
        return task

    def get_sample_size(self, client_config=None, task: ClientTask = None):
        if client_config is None:
            client_config = self.client_config
        if task is not None:
            task.task_execution_start_time = time.time()
            sample_size = get_sample_size_executor(client_agent_config=client_config)
            task.result = sample_size
            task.task_execution_finish_time = time.time()
            return task
        else:
            return get_sample_size_executor(client_agent_config=client_config)

    def data_readiness_report(self, client_config=None, task: ClientTask = None):
        if client_config is None:
            client_config = self.client_config
        if task is not None:
            task.task_execution_start_time = time.time()
            data_readiness_report = data_readiness_report_executor(client_agent_config=client_config)
            task.result = data_readiness_report
            task.task_execution_finish_time = time.time()
            return task
        else:
            return data_readiness_report_executor(client_agent_config=client_config)

    def train(self, model_ref, metadata=None, client_config=None, task: ClientTask = None):
        if client_config is None:
            client_config = self.client_config
        if isinstance(model_ref, ray.ObjectRef):
            model = ray.get(model_ref)
        else:
            model = model_ref

        if task is not None:
            task.task_execution_start_time = time.time()
            train_res = train_executor(
                client_agent_config=client_config,
                model=model,
                meta_data=metadata,
            )
            task.result = train_res
            task.task_execution_finish_time = time.time()
            return task
        else:
            return train_executor(
                client_agent_config=client_config,
                model=model,
                meta_data=metadata,
            )
