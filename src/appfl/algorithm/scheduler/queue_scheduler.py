import math
import threading
from omegaconf import DictConfig
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Union, Dict, Tuple, List
from appfl.algorithm.scheduler import BaseScheduler
from appfl.algorithm.aggregator import BaseAggregator

class QueueScheduler(BaseScheduler):
    """
    Scheduler for `FedQueue` asynchronous federated learning algorithm.
    """
    def __init__(
        self,
        scheduler_configs: DictConfig,
        aggregator: BaseAggregator,
        logger: Any
    ):
        super().__init__(scheduler_configs, aggregator, logger)
        self.global_round = 0
        self.num_clients = self.scheduler_configs.num_clients
        self.t_sync = self.scheduler_configs.get("t_sync", 3600)
        self.lr_base = self.scheduler_configs.get("lr_base", 0.01)
        self.warm_up_steps = self.scheduler_configs.get("warm_up_steps", 20)
        self.alpha_queue = self.scheduler_configs.get("alpha_queue", 0.5)
        self.alpha_compute = self.scheduler_configs.get("alpha_compute", 0.5)
        self.safety_buffer = self.scheduler_configs.get("safety_buffer", 60)
        self.admission_tolerance = self.scheduler_configs.get("admission_tolerance", 0.15)
        self._access_lock = threading.Lock()
        self.future_record = {}
        self.timer_record = None
        self.client_round_record = {}
        self.client_steps_record = {}
        self.queue_time_estimation = {}
        self.compute_time_estimation = {}
        self.client_model_buffer = self._reset_client_model_buffer()

    def get_parameters(
        self, **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        with self._access_lock:
            return super().get_parameters(**kwargs)

    def schedule(
        self,
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict],
        **kwargs,
    ) -> Future:
        with self._access_lock:
            self._update_queue_estimation(client_id, kwargs)
            self._update_compute_estimation(client_id, kwargs)
            self.client_model_buffer["local_models"][client_id] = local_model
            self.client_model_buffer["local_steps"][client_id] = self._get_local_steps(client_id)
            self.client_model_buffer["curr_round"][client_id] = self._get_curr_round(client_id)
            future = Future()
            self.future_record[client_id] = future
            if len(self.client_model_buffer["local_models"]) == self.num_clients:
                self._aggregate_global_model(lock_acquired=True)
            return future
    
    def _aggregate_global_model(
        self,
        lock_acquired: bool = False,
    ) -> None:
        if not lock_acquired:
            self._access_lock.acquire()
            
        all_clients = list(self.client_model_buffer["local_models"].keys())

        if len(all_clients) > 0:
            staleness = {
                client_id: self.global_round - self.client_model_buffer["curr_round"][client_id]
                for client_id in all_clients
            }
            global_model = self.aggregator.aggregate(
                local_models=self.client_model_buffer["local_models"],
                staleness=staleness,
                local_steps=self.client_model_buffer["local_steps"], # check this for computing p_k
            )
            self.global_round += 1
            for client_id in all_clients:
                client_metadata = self._get_client_metadata(client_id, all_clients)
                self.future_record[client_id].set_result((global_model, client_metadata))
                self.client_round_record[client_id] = self.global_round
                del self.future_record[client_id]
            self._reset_client_model_buffer()
            
        if lock_acquired:
            if self.timer_record is not None:
                self.timer_record.stop()
                self.timer_record = None
        
        timer = threading.Timer(
            (1 + self.admission_tolerance) * self.t_sync,
            self._aggregate_global_model,
            {"lock_acquired": False}
        )
        timer.start()

        if not lock_acquired:
            self._access_lock.release()

    def _update_queue_estimation(
        self, 
        client_id: Union[int, str], 
        kwargs: Dict,
    ) -> None:
        assert "queue_time" in kwargs, "QueueScheduler requires `queue_time` in kwargs."
        queue_time = kwargs["queue_time"]
        if client_id not in self.queue_time_estimation:
            self.queue_time_estimation[client_id] = queue_time
        self.queue_time_estimation[client_id] = self.alpha_queue * queue_time + \
            (1 - self.alpha_queue) * self.queue_time_estimation[client_id]
        
    def _update_compute_estimation(
        self,
        client_id: Union[int, str],
        kwargs: Dict,
    ) -> None:
        assert "compute_second_per_step" in kwargs, "QueueScheduler requires `compute_second_per_step` in kwargs."
        compute_second_per_step = kwargs["compute_second_per_step"]
        if client_id not in self.compute_time_estimation:
            self.compute_time_estimation[client_id] = compute_second_per_step
        self.compute_time_estimation[client_id] = self.alpha_compute * compute_second_per_step + \
            (1 - self.alpha_compute) * self.compute_time_estimation[client_id]

    def _get_local_steps(
        self,
        client_id: Union[int, str]
    ) -> int:
        if not client_id in self.client_steps_record:
            self.client_steps_record[client_id] = self.warm_up_steps
        return self.client_steps_record[client_id]
    
    def _get_curr_round(
        self,
        client_id: Union[int, str]
    ) -> int:
        if not client_id in self.client_round_record:
            self.client_round_record[client_id] = 0
        return self.client_round_record[client_id]

    def _reset_client_model_buffer(self):
        self.client_model_buffer = {
            "local_models": {},
            "local_steps": {},
            "curr_round": {}
        }

    def _get_client_metadata(
        self,
        client_id: Union[int, str],
        all_clients: List[Union[int, str]]
    ) -> Dict[str, Any]:
        job_budget = self.t_sync - self.queue_time_estimation[client_id] - self.safety_buffer
        local_steps = math.floor(job_budget / self.compute_time_estimation[client_id])
        all_local_steps = [
            math.floor(self.t_sync - self.queue_time_estimation[cid] - self.safety_buffer) / 
            self.compute_time_estimation[cid] 
            for cid in all_clients
        ]
        min_local_steps = min(all_local_steps)
        learning_rate = self.lr_base * (min_local_steps / local_steps)
        client_metadata = {
            "job_budget": job_budget,
            "local_steps": local_steps,
            "learning_rate": learning_rate
        }
        self.client_steps_record[client_id] = local_steps
        return client_metadata
