import ray
import uuid
import time
from omegaconf import OmegaConf
from ray.exceptions import RayActorError
import ray.util.state as state_api
import math
import threading
import asyncio

from appfl.logger import ServerAgentFileLogger
from appfl.comm.ray import RayClientCommunicator
from appfl.comm.base import BaseServerCommunicator
from appfl.comm.utils.config import ClientTask
from appfl.comm.utils.client_info import ClientInfo
from appfl.comm.utils.s3_storage import CloudStorage, LargeObjectWrapper
from appfl.config import ServerAgentConfig, ClientAgentConfig
from typing import List, Optional, Union, Dict, OrderedDict, Any, Tuple
from ray.autoscaler.sdk import terminate_node_by_resource_tag


class RayCostAwareServerCommunicator(BaseServerCommunicator):
    def __init__(
        self,
        server_agent_config: ServerAgentConfig,
        client_agent_configs: List[ClientAgentConfig],
        clients_info: Dict[str, ClientInfo],
        logger: Optional[ServerAgentFileLogger] = None,
        **kwargs,
    ):
        self.comm_type = "ray-costaware"
        super().__init__(server_agent_config, client_agent_configs, logger, **kwargs)
        ray.init(address="auto")
        # TODO pick value from server config
        self.TERMINATION_BUFFER_SEC = 30
        self.clients_info = clients_info
        self.client_actors = {}
        _client_id_check_set = set()
        self.client_configs = {}
        for client_config in client_agent_configs:
            assert hasattr(client_config, "client_id"), (
                "Client configuration must have an client_id."
            )
            client_id = client_config["client_id"]
            assert client_id not in _client_id_check_set, (
                f"Client ID {client_id} is not unique for this client configuration.\n{client_config}"
            )
            _client_id_check_set.add(client_id)
            # Read the client dataloader source file
            with open(client_config.data_configs.dataset_path) as file:
                client_config.data_configs.dataset_source = file.read()
            del client_config.data_configs.dataset_path

            client_config.experiment_id = self.experiment_id
            client_config = OmegaConf.merge(
                server_agent_config.client_configs, client_config
            )
            client_config.comm_configs.comm_type = self.comm_type
            self.clients_info[client_id].batch_size = client_config.train_configs.get(
                "train_batch_size", 32
            )
            self.client_configs[client_id] = client_config
            assert (
                not self.__is_checkpointing_enabled(client_config) or self.use_s3bucket
            ), "For checkpointing to work you are required to enable s3"

            # If specific client is required to use specific resource type, i.e,  `assign_random = False`
            if (
                hasattr(server_agent_config.server_configs, "comm_configs")
                and hasattr(
                    server_agent_config.server_configs.comm_configs, "ray_configs"
                )
                and not server_agent_config.server_configs.comm_configs.ray_configs.get(
                    "assign_random", False
                )
            ):
                if (
                    hasattr(client_config.train_configs, "device")
                    and "cuda" in client_config.train_configs.device
                ):
                    self.client_actors[client_id] = RayClientCommunicator.options(
                        resources={client_config["client_id"]: 1}, num_gpus=1
                    ).remote(server_agent_config, client_config)
                else:
                    self.client_actors[client_id] = RayClientCommunicator.options(
                        resources={client_config["client_id"]: 1}
                    ).remote(server_agent_config, client_config)

            else:
                self.client_actors[client_id] = RayClientCommunicator.remote(
                    server_agent_config, client_config
                )

        self.executing_tasks: Dict[str, ClientTask] = {}
        self.executing_task_futs: Dict[Any, str] = {}
        self.clients_futs: Dict[str, Any] = {}
        self.budget_updated_till = -1
        self.spinup_queue = []
        self.clients_to_terminate = []

        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.monitor_thread.start()

    def send_task_to_all_clients(
        self,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Union[Dict, List[Dict]] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to all clients.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket and model is not None:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            model = CloudStorage.upload_object(model_wrapper, register_for_clean=True)
        for i, client_id in enumerate(self.client_actors):
            client_metadata = metadata[i] if isinstance(metadata, list) else metadata
            if need_model_response and self.use_s3bucket:
                local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
                local_model_url = CloudStorage.presign_upload_object(local_model_key)
                client_metadata["local_model_key"] = local_model_key
                client_metadata["local_model_url"] = local_model_url
            is_actor_alive = self._check_actor_state(client_id)
            task_id, task_ref = self.__send_task(
                self.client_actors[client_id],
                task_name,
                model,
                client_metadata,
                client_id,
                is_actor_alive,
            )
            self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def send_task_to_one_client(
        self,
        client_id: str,
        task_name: str,
        *,
        model: Optional[Union[Dict, OrderedDict, bytes]] = None,
        metadata: Optional[Dict] = {},
        need_model_response: bool = False,
    ):
        """
        Send a specific task to one specific client.
        :param `client_id`: The client id to which the task is sent.
        :param `task_name`: Name of the task to be executed on the clients
        :param [Optional] `model`: Model to be sent to the clients
        :param [Optional] `metadata`: Additional metadata to be sent to the clients
        :param `need_model_response`: Whether the task requires a model response from the clients
            If so, the server will provide a pre-signed URL for the clients to upload the model if using S3.
        """
        if self.use_s3bucket and model is not None:
            model_wrapper = LargeObjectWrapper(
                data=model,
                name=str(uuid.uuid4()) + "_server_state",
            )
            model = CloudStorage.upload_object(model_wrapper, register_for_clean=True)
        if need_model_response and self.use_s3bucket:
            local_model_key = f"{str(uuid.uuid4())}_client_state_{client_id}"
            local_model_url = CloudStorage.presign_upload_object(local_model_key)
            metadata["local_model_key"] = local_model_key
            metadata["local_model_url"] = local_model_url
        is_actor_alive = self._check_actor_state(client_id)
        task_id, task_ref = self.__send_task(
            self.client_actors[client_id],
            task_name,
            model,
            metadata,
            client_id,
            is_actor_alive,
        )
        self.logger.info(f"Task '{task_name}' is assigned to {client_id}.")

    def _check_actor_state(self, client_id: str):
        ray_actors = ray._private.state.actors()
        ray_actor_details = ray_actors[
            self.client_actors[client_id]._ray_actor_id.hex()
        ]
        if ray_actor_details["State"] == "DEAD":
            if "cuda" in self.client_configs[client_id].train_configs.get("device", "cpu"):
                self.client_actors[client_id] = RayClientCommunicator.options(
                    resources={client_id: 1}, num_gpus=1
                ).remote(self.server_agent_config, self.client_configs[client_id])
            else:
                self.client_actors[client_id] = RayClientCommunicator.options(
                    resources={client_id: 1}
                ).remote(self.server_agent_config, self.client_configs[client_id])
            return False
        return True

    def recv_result_from_all_clients(self) -> Tuple[Dict, Dict]:
        """
        Receive task results from all clients that have running tasks.
        :return `client_results`: A dictionary containing the results from all clients - Dict[client_id, client_model]
        :return `client_metadata`: A dictionary containing the metadata from all clients - Dict[client_id, client_metadata]
        """
        client_results, client_metadata, client_task_info = {}, {}, {}
        while len(self.executing_task_futs):
            try:
                done_futs, _ = ray.wait(
                    list(self.executing_task_futs.keys()), num_returns=1, timeout=None
                )

                if not done_futs:
                    continue

                fut = done_futs[0]
                task_id = self.executing_task_futs[fut]
                client_id = self.executing_tasks[task_id].client_id

                result = ray.get(fut)
                self._update_spinup_time(result)
                self._update_epoch_estimate_time(result)
                client_model, client_metadata_local = self._parse_result(result.result)
                self._update_learning_rate(result, client_metadata_local)
                client_metadata_local = self._check_deprecation(
                    client_id, client_metadata_local
                )
                client_results[client_id] = client_model
                client_metadata[client_id] = client_metadata_local
                client_task_info[client_id] = result
                result.result = None
                self.clients_info[client_id].tasks.append(result)
                self.executing_tasks[task_id] = result
                self.__update_executing_task(
                    client_metadata_local, task_id, client_id, fut
                )

            except RayActorError as e:
                self.logger.error(f"Client stopped with exception: {str(e)}")
                self.__relaunch_client_actor(e)

        return client_results, client_metadata, client_task_info

    def recv_result_from_one_client(self) -> Tuple[str, Any, Dict]:
        """
        Receive task results from the first client that finishes the task.
        :return `client_id`: The client id from which the result is received.
        :return `client_model`: The model returned from the client
        :return `client_metadata`: The metadata returned from the client
        """
        assert len(self.executing_task_futs), "There is no active client running tasks."
        try:
            ready_refs, _ = ray.wait(
                list(self.executing_task_futs), num_returns=1, timeout=None
            )
            finished_ref = ready_refs[0]
            task_id = self.executing_task_futs[finished_ref]
            result = ray.get(finished_ref)
            self._update_spinup_time(result)
            self._update_epoch_estimate_time(result)
            client_id = self.executing_tasks[task_id].client_id
            client_model, client_metadata = self._parse_result(result.result)
            self._update_learning_rate(result, client_metadata)
            client_metadata = self._check_deprecation(client_id, client_metadata)
            result.result = None
            self.clients_info[client_id].tasks.append(result)
            self.executing_tasks[task_id] = result
            self.__update_executing_task(
                client_metadata, task_id, client_id, finished_ref
            )
            if result.task_name == "spinup":
                return self.recv_result_from_one_client()

            return client_id, client_model, client_metadata, result

        except RayActorError as e:
            self.logger.error(f"Client stopped with exception: {str(e)}")
            self.__relaunch_client_actor(e)
            return self.recv_result_from_one_client()

    def shutdown_all_clients(self):
        """Cancel all the running tasks on the clients and shutdown the globus compute executor."""
        self.stop_resource_monitor()
        self.logger.info("Shutting down all clients......")
        ray.shutdown()
        # Clean-up cloud storage
        if self.use_s3bucket:
            CloudStorage.clean_up()
        # Clean-up proxystore
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except Exception:  # noqa: E722
                self.proxystore.close()
        self.logger.info(
            "The server and all clients have been shutted down successfully."
        )

    def cancel_all_tasks(self):
        """Cancel all on-the-fly client tasks."""
        for task_fut in self.executing_task_futs:
            ray.cancel(task_fut)
            task_id = self.executing_task_futs[task_fut]
            client_id = self.executing_tasks[task_id].client_id
            self.logger.info(f"cancelled task id: {task_id}, client id: {client_id}")
        self.executing_task_futs = {}
        self.executing_tasks = {}

    def terminate_all_clients(self):
        resource_tag = {}
        for client_id in self.clients_info.keys():
            client_info = self.clients_info[client_id]
            client_info.instance_alive = False
            self.logger.info(f"Terminating instance of client {client_id}")
            resource_tag[client_id] = 1
        terminate_node_by_resource_tag(resource_tag)
        time.sleep(45)

    def warmup_clients(self, clients_sample_size_dict: Dict[str, int], model):
        """
        Estimates per epoch time by sending task with few steps to clients.
        One step warmup: if trainer code returns metadata with training time excluding overhead like preval and postval.
            we send 50% steps of total steps of client with least data to clients to get an estimate for an epoch
        Two step warmup: if trainer code does not returns metadata we estimate overhead and training time by sending
            two warmup task 1 step and 50% steps of total steps of client with least data.
            By doing math we can find actual training time and extrapolate it for an epoch.
        Args:
            clients_sample_size_dict: Dict[str, int]
            model:
        """
        # update sample size and steps_per_epoch in clients info
        # get client with minimum sample size
        two_step_warmup = False
        min_sample_size = 999999999
        min_data_client_id = None
        for client_id in clients_sample_size_dict.keys():
            sample_size = clients_sample_size_dict[client_id]
            batch_size = self.clients_info[client_id].batch_size
            self.clients_info[client_id].sample_size = sample_size
            self.clients_info[client_id].steps_per_epoch = math.ceil(
                sample_size / (batch_size * 1.0)
            )
            if sample_size < min_sample_size:
                min_sample_size = sample_size
                min_data_client_id = client_id

        self.logger.info(
            f"client {min_data_client_id} with least sample size of {min_sample_size}"
        )

        if two_step_warmup:
            # 0% step size
            round_one_steps, round_one_train_res, _ = self._get_warmup_training_res(
                min_data_client_id, 0, model
            )
            # 50% step size
            round_two_steps, round_two_train_res, _ = self._get_warmup_training_res(
                min_data_client_id, 50, model
            )

            self.logger.info(f"round one results {round_one_train_res}")
            self.logger.info(f"round two results {round_two_train_res}")
            self._update_epoch_est_after_warmup(
                two_step_warmup,
                round_one_steps,
                round_one_train_res,
                round_two_steps,
                round_two_train_res,
            )
        else:
            round_one_steps, round_one_train_res, train_metadata = (
                self._get_warmup_training_res(min_data_client_id, 100, model)
            )
            self._update_epoch_est_after_warmup(
                two_step_warmup,
                round_one_steps,
                round_one_train_res,
                train_metadata=train_metadata,
            )

    def check_client_termination_and_schedule_spinup(self, client_id):
        """
        Checks if client instance need to be terminated, based on the slowest task estimated time to finish the task.
        If we have enough time for spinup we terminate the client instance and schedule its spinup time in the queue.
        Args:
            client_id:
        """
        max_task = max(
            (
                task
                for task in self.executing_tasks.values()
                if task.task_name == "train"
            ),
            key=lambda task: task.est_finish_time,
            default=None,
        )
        if max_task is not None:
            max_time = max_task.est_finish_time
            current_time = time.time()
            if current_time > max_time:
                self.logger.info(
                    f"slowest task should have been finished according to estimate {max_task.client_id}"
                )
                return
            if (
                max_time - current_time
                > self.clients_info[client_id].est_spinup_time
                + self.TERMINATION_BUFFER_SEC
            ):
                self.clients_to_terminate.append(client_id)
                client_info = self.clients_info[client_id]
                if (
                    client_info.budget
                    < client_info.est_time_per_epoch
                    * client_info.spot_price_per_hr
                    / 3600
                ):
                    client_info.inactive = True
                else:
                    # set spinup time
                    self.spinup_queue.append(
                        (
                            client_id,
                            max_time
                            - self.clients_info[client_id].est_spinup_time
                            - self.TERMINATION_BUFFER_SEC,
                        )
                    )

    # def spinup_clients(self):
    #     for client_id in self.clients_info.keys():
    #         client_info = self.clients_info[client_id]
    #         if not client_info.inactive and not client_info.instance_triggered:
    #             self.server_communicator.send_task_to_one_client(
    #                 client_id,
    #                 task_name="spinup",
    #             )
    #     self.recv_result_from_all_clients()

    def _update_epoch_est_after_warmup(
        self,
        two_step_warmup,
        round_one_steps,
        round_one_train_res: Dict[str, ClientTask],
        round_two_steps=None,
        round_two_train_res: Dict[str, ClientTask] = None,
        train_metadata=None,
    ):
        """Method to update epoch estimated time after warmup completion"""
        self.logger.info("Updating estimated epoch time after warmup for each client.")
        if two_step_warmup:
            for client_id in round_one_train_res.keys():
                round_one_total_time = int(
                    round_one_train_res[client_id].task_execution_finish_time
                    - round_one_train_res[client_id].task_execution_start_time
                )
                round_two_total_time = int(
                    round_two_train_res[client_id].task_execution_finish_time
                    - round_two_train_res[client_id].task_execution_start_time
                )
                self.logger.info(
                    f"{client_id} round one total time: {round_one_total_time}"
                )
                self.logger.info(
                    f"{client_id} round two total time: {round_two_total_time}"
                )
                time_per_step = (round_two_total_time - round_one_total_time) / (
                    round_two_steps - round_one_steps
                )
                # average overhead, includes pre validation and post training validation
                train_time_overhead_sec = (
                    (round_two_total_time - time_per_step * round_two_steps)
                    + (round_one_total_time - time_per_step * round_one_steps)
                ) / 2
                est_time_per_epoch = (
                    time_per_step * self.clients_info[client_id].steps_per_epoch
                    + train_time_overhead_sec
                )
                self.clients_info[
                    client_id
                ].train_time_overhead_sec = train_time_overhead_sec
                self.clients_info[client_id].time_per_step = time_per_step
                self.clients_info[client_id].est_time_per_epoch = est_time_per_epoch
                self.logger.info(
                    f"Client {client_id} estimated time per epoch: {est_time_per_epoch}"
                )
        else:
            for client_id in round_one_train_res.keys():
                round_one_total_time = int(
                    round_one_train_res[client_id].task_execution_finish_time
                    - round_one_train_res[client_id].task_execution_start_time
                )
                training_time = train_metadata[client_id]["per_step_time"]
                train_time_overhead_sec = round_one_total_time - training_time
                time_per_step = training_time / round_one_steps
                est_time_per_epoch = (
                    time_per_step * self.clients_info[client_id].steps_per_epoch
                    + train_time_overhead_sec
                )
                self.clients_info[
                    client_id
                ].train_time_overhead_sec = train_time_overhead_sec
                self.clients_info[client_id].time_per_step = time_per_step
                self.clients_info[client_id].est_time_per_epoch = est_time_per_epoch
                self.logger.info(
                    f"Client {client_id} estimated time per epoch: {est_time_per_epoch}"
                )

    def _get_warmup_training_res(self, min_data_client_id, percent_step, model):
        """updates clients config to send warmup task to all the clients and returns back its results"""
        num_of_steps = math.ceil(
            percent_step / 100 * self.clients_info[min_data_client_id].steps_per_epoch
        )
        if num_of_steps == 0:
            num_of_steps = 1
        for client_id in self.client_configs.keys():
            self.client_configs[client_id].train_configs["mode"] = "step"
            self.client_configs[client_id].train_configs["num_local_steps"] = (
                num_of_steps
            )

        self.logger.info(
            f"Sending warmup task to all clients for num_local_steps: {num_of_steps}"
        )
        self.send_task_to_all_clients("train", model=model, need_model_response=True)

        res = self.recv_result_from_all_clients()
        metadata = res[1]
        train_res = res[2]
        return num_of_steps, train_res, metadata

    def _update_learning_rate(self, task: ClientTask, client_metadata):
        """After receiving result of training task we update the learning rate for clients"""
        client_id = task.client_id
        if (
            task.task_name == "train"
            and self.client_configs[client_id].train_configs["mode"] == "epoch"
            and hasattr(self.client_configs[client_id].train_configs, "optim_args")
            and hasattr(self.client_configs[client_id].train_configs.optim_args, "lr")
        ):
            self.client_configs[client_id].train_configs.optim_args["lr"] = (
                client_metadata["next_round_lr"]
            )
            self.logger.info(
                f"Updated learning rate to {client_metadata['next_round_lr']} for Client {client_id}"
            )

    def _update_epoch_estimate_time(self, task: ClientTask):
        """After receiving result of training task we update the epoch estimate time for clients"""
        client_id = task.client_id
        if (
            task.task_name == "train"
            and self.client_configs[client_id].train_configs["mode"] == "epoch"
        ):
            total_execution_time = int(
                task.task_execution_finish_time - task.task_execution_start_time
            )
            alpha = self.clients_info[client_id].alpha
            if not task.is_instance_alive or (
                len(self.clients_info[client_id].tasks) > 0
                and self.clients_info[client_id].tasks[-1].task_name == "spinup"
            ):
                if self.clients_info[client_id].est_time_per_epoch_after_spinup == -1:
                    self.clients_info[
                        client_id
                    ].est_time_per_epoch_after_spinup = total_execution_time
                else:
                    self.clients_info[client_id].est_time_per_epoch_after_spinup = (
                        1 - alpha
                    ) * self.clients_info[client_id].est_time_per_epoch_after_spinup + (
                        alpha * total_execution_time
                    )
                self.logger.info(
                    f"Updated est_time_per_epoch_after_spinup to {self.clients_info[client_id].est_time_per_epoch_after_spinup} for Client {client_id}"
                )
            else:
                if self.clients_info[client_id].est_time_per_epoch == -1:
                    self.clients_info[
                        client_id
                    ].est_time_per_epoch = total_execution_time
                else:
                    self.clients_info[client_id].est_time_per_epoch = (
                        1 - alpha
                    ) * self.clients_info[client_id].est_time_per_epoch + (
                        alpha * total_execution_time
                    )
                self.logger.info(
                    f"Updated est_time_per_epoch to {self.clients_info[client_id].est_time_per_epoch} for Client {client_id}"
                )

    def _update_spinup_time(self, task: ClientTask):
        """After receiving result we check if a spinup took place and if it does we update the spinup estimate time for client"""
        client_id = task.client_id
        task.task_execution_time = int(task.task_execution_finish_time) - int(
            task.task_execution_start_time
        )
        # nodes_info = state_api.list_nodes(detail=True)
        # node_info = self.__get_current_client_node_info(nodes_info, client_id)
        # if node_info is None:
        #     return
        # node start after task was submitted, it means new node was needed
        if int(task.task_execution_start_time) - int(task.start_time) > 80:
            spinup_time = int(task.task_execution_start_time) - int(task.start_time)
            task.spin_up_time = spinup_time
            if (
                len(self.clients_info[client_id].tasks) > 0
                and self.clients_info[client_id].tasks[-1].task_name == "spinup"
            ):
                return
            # using execution_start_time instead of node start time as we need to take all overhead in consideration for pre start
            self.logger.info(
                f"Updating spinup time for client {client_id} to {spinup_time}"
            )
            if self.clients_info[client_id].est_spinup_time == -1:
                self.clients_info[client_id].est_spinup_time = spinup_time
            else:
                alpha = self.clients_info[client_id].alpha
                self.clients_info[client_id].est_spinup_time = (
                    1 - alpha
                ) * self.clients_info[client_id].est_spinup_time + (alpha * spinup_time)

    def __get_current_client_node_info(self, nodes_info, client_id):
        for node_info in nodes_info:
            if (
                node_info.state == "ALIVE"
                and client_id in node_info.resources_total.keys()
            ):
                return node_info

    def _run_async_loop(self):
        """Runs the asyncio event loop in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._monitor())

    def stop_resource_monitor(self):
        """Stops the background resource monitoring thread."""
        self.stop_event.set()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()  # Wait for the thread to exit safely
        print("monitoring stopped.")

    async def _monitor(self):
        """Daemon process that updates the budget of the clients based on usage and also spins up the clients that are scheduled"""
        while not self.stop_event.is_set():
            await asyncio.to_thread(self._update_budget)
            # TODO spin up clients
            for i in range(30):  # Check the stop_event every second
                if i % 15 == 0:
                    await asyncio.to_thread(self._spinup_clients)
                    await asyncio.to_thread(self._terminate_clients)
                if self.stop_event.is_set():
                    return
                time.sleep(1)
            # await asyncio.sleep(30)

    def _terminate_clients(self):
        # we are terminating in async manner using queue as interaction with autoscaler is through shared kv and
        # termination happens in async manner so we collect all the nodes to terminate and then terminate it in a batch
        resource_tag = {}
        for client_id in self.clients_to_terminate:
            client_info = self.clients_info[client_id]
            client_info.instance_alive = False
            self.logger.info(f"Terminating instance of client {client_id}")
            resource_tag[client_id] = 1
        if len(resource_tag) > 0:
            terminate_node_by_resource_tag(resource_tag)
            self.clients_to_terminate = []

    def _update_budget(self):
        """It keeps the track of the time till which it has updated budget of the clients and then using the rays API checks uptime of instances and updates their respective budgets"""
        nodes_info = state_api.list_nodes(detail=True)
        update_time = int(time.time())
        client_node_info = {client_id: [] for client_id in self.client_configs.keys()}
        alive_instance_client_ids = set()
        for node_info in nodes_info:
            if not node_info.is_head_node and (
                node_info.end_time_ms == 0
                or int(node_info.end_time_ms // 1000) > self.budget_updated_till
            ):
                for key in node_info.resources_total.keys():
                    if key in self.client_configs.keys():
                        client_node_info[key].append(node_info)
                        if node_info.end_time_ms == 0:
                            alive_instance_client_ids.add(key)

        for client_id in client_node_info:
            client_node_info[client_id].sort(key=lambda x: x["start_time_ms"])
            # maintain state of the clients instance
            if client_id in alive_instance_client_ids:
                self.clients_info[client_id].instance_triggered = False
                self.clients_info[client_id].instance_triggered_at = -1
                self.clients_info[client_id].instance_alive = True
                self.logger.info(f"Client {client_id} instance is running")
            else:
                self.clients_info[client_id].instance_alive = False
                self.logger.info(f"Client {client_id} instance is not running")

        for client_id in client_node_info:
            for node_info in client_node_info[client_id]:
                start = max(node_info.start_time_ms // 1000, self.budget_updated_till)
                end = node_info.end_time_ms // 1000
                if end == 0:
                    end = update_time
                time_in_sec = end - start
                print(f"Updating budget for client {client_id} time {time_in_sec}")
                self.clients_info[client_id].budget -= (
                    time_in_sec
                    * self.clients_info[client_id].spot_price_per_hr
                    / 3600.0
                )
                self.clients_info[client_id].total_instance_up_time += time_in_sec
        self.budget_updated_till = update_time

    def _spinup_clients(self):
        """Check the spinup queue and spinup the client who are ready"""
        current_time = time.time()
        # Iterate in reverse to safely remove items by index
        for i in reversed(range(len(self.spinup_queue))):
            client_id, spinup_time = self.spinup_queue[i]
            if spinup_time < current_time:
                client_info = self.clients_info[client_id]
                if (
                    not client_info.inactive
                    and not client_info.instance_triggered
                    and not client_info.instance_alive
                ):
                    self.send_task_to_one_client(
                        client_id,
                        task_name="spinup",
                    )
                    client_info.instance_triggered = True
                    client_info.instance_triggered_at = current_time
                self.spinup_queue.pop(i)

    @staticmethod
    def __is_checkpointing_enabled(client_config):
        if hasattr(client_config, "comm_configs") and hasattr(
            client_config.comm_configs, "checkpoint_configs"
        ):
            return client_config.comm_configs.checkpoint_configs.get(
                "enable_checkpointing", False
            )
        return False

    def __relaunch_client_actor(self, exception):
        """Relaunch the client if any of the client instance dies"""
        actor_id = str(exception.actor_id)
        interrupted_client_id = None
        # find the interrupted client id using the actor id
        for client_id in self.client_actors.keys():
            if actor_id == str(self.client_actors[client_id]._ray_actor_id.hex()):
                interrupted_client_id = client_id
                break
        self.logger.info(f"Client: {interrupted_client_id} execution was interrupted")
        if not self.__is_checkpointing_enabled(
            self.client_configs[interrupted_client_id]
        ):
            self.__remove_old_tasks(interrupted_client_id, False)
            return
        self.logger.info(f"Relaunching client {interrupted_client_id}")
        # re launch the ray client
        if "cuda" in self.client_configs[interrupted_client_id].train_configs.get("device", "cpu"):
            self.client_actors[interrupted_client_id] = RayClientCommunicator.options(
                resources={interrupted_client_id: 1}, num_gpus=1
            ).remote(self.server_agent_config, self.client_configs[interrupted_client_id])
        else:
            self.client_actors[interrupted_client_id] = RayClientCommunicator.options(
                resources={interrupted_client_id: 1}
            ).remote(self.server_agent_config, self.client_configs[interrupted_client_id])
        # check all the ObjectRef which needs rerun on the new client
        self.__remove_old_tasks(interrupted_client_id, True)

    def __update_spinup_time_on_exception(self, client_id):
        """During exception we relaunch the tasks to clients, this method makes sure that already scheduled tasks spinup times are updated in queue according to the slowest task in queue"""
        max_task = max(
            self.executing_tasks.values(),
            key=lambda task: task.est_finish_time,
            default=None,
        )
        # check if the slowest task is the client which had exception, if yes update spinup time for rest of the queued tasks
        if max_task is not None and max_task.client_id == client_id:
            max_time = max_task.est_finish_time
            for client_id, _ in self.spinup_queue:
                self.spinup_queue.append(
                    (client_id, max_time - self.clients_info[client_id].est_spinup_time)
                )

    def __remove_old_tasks(self, client_id, trigger_again=False):
        """
        removes the old task from the executing queue and if asked for retrigger queues the task to the new spawned actor
        """
        for old_fut in self.clients_futs[client_id]:
            task_id = self.executing_task_futs[old_fut]
            client_task = self.executing_tasks[task_id]
            client_task.failure = True
            client_task.pending = False
            # send the old task to new client
            if trigger_again:
                self.logger.info(f"retriggering task {task_id} {str(client_task)}")
                # model = OmegaConf.to_container(
                #     client_task.parameters["model"], resolve=True
                # )
                # metadata = OmegaConf.to_container(
                #     client_task.parameters["metadata"], resolve=True
                # )
                self.__send_task(
                    self.client_actors[client_id],
                    client_task.task_name,
                    client_task.parameters["model"],
                    client_task.parameters["metadata"],
                    client_id,
                    is_actor_alive=False,
                )
            # remove old tasks from the queue
            self.executing_tasks.pop(task_id)
            self.executing_task_futs.pop(old_fut)
            self.clients_futs[client_id].remove(old_fut)
        self.__update_spinup_time_on_exception(client_id)

    def __update_executing_task(
        self, client_metadata_local, task_id, client_id, task_fut
    ):
        try:
            self.executing_tasks[task_id].end_time = time.time()
            self.executing_tasks[task_id].success = True
            self.executing_tasks[task_id].pending = False
            self.executing_tasks[task_id].log = client_metadata_local.get("log", {})
            # Clean up the task
            self.logger.info(
                f"Received results of task '{self.executing_tasks[task_id].task_name}' from {client_id} took time {self.executing_tasks[task_id].task_execution_finish_time - self.executing_tasks[task_id].task_execution_start_time}."
            )
            self.executing_tasks.pop(task_id)
            self.executing_task_futs.pop(task_fut)
            self.clients_futs[client_id].remove(task_fut)
        except Exception as e:
            self.logger.error(
                f"Task {self.executing_tasks[task_id].task_name} on {client_id} failed with an error."
            )
            raise e

    def __send_task(
        self,
        client: RayClientCommunicator,
        task_name,
        model,
        metadata,
        client_id,
        is_actor_alive,
    ):
        task_id = str(uuid.uuid4())
        task = ClientTask(
            task_id=task_id,
            task_name=task_name,
            client_id=client_id,
            start_time=time.time(),
        )
        task.is_instance_alive = is_actor_alive
        ref = None
        current_time = time.time()
        if task_name == "get_sample_size":
            ref = client.get_sample_size.remote(self.client_configs[client_id], task)
        elif task_name == "data_readiness_report":
            ref = client.data_readiness_report.remote(
                self.client_configs[client_id], task
            )
        elif task_name == "train":
            if self.client_configs[client_id].train_configs["mode"] == "epoch":
                client_info = self.clients_info[client_id]
                if client_info.instance_triggered:
                    task.est_finish_time = (
                        current_time
                        + max(
                            client_info.est_spinup_time
                            - (current_time - client_info.instance_triggered_at),
                            0,
                        )
                        + client_info.est_time_per_epoch_after_spinup
                    )
                elif not client_info.instance_alive:
                    client_info.instance_triggered = True
                    client_info.instance_triggered_at = current_time
                    task.est_finish_time = (
                        current_time
                        + client_info.est_time_per_epoch_after_spinup
                        + client_info.est_spinup_time
                    )
                elif client_info.instance_alive:
                    task.est_finish_time = current_time + client_info.est_time_per_epoch
            ref = client.train.remote(
                model, metadata, self.client_configs[client_id], task
            )
        elif task_name == "spinup":
            ref = client.spinup.remote(task)

        self.executing_tasks[task_id] = task
        self.executing_task_futs[ref] = task_id

        if self.__is_checkpointing_enabled(self.client_configs[client_id]):
            parameters = {"model": model, "metadata": metadata}
            self.executing_tasks[task_id].parameters = parameters
        if client_id in self.clients_futs.keys():
            self.clients_futs[client_id].append(ref)
        else:
            self.clients_futs[client_id] = [ref]

        return task_id, ref
