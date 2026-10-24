APPFL Server Agent
==================

Functionalities
---------------

The APPFL server agent acts on behalf of the federated learning server to fulfill various requests from the clients, including:

- Get the current model parameters (useful for initialization and for getting the final trained model)
- Get the configurations for local training and other tasks that are shared among all clients (e.g., trainer type, compression configurations, model architecture, etc.)
- Update the global model using the model parameters/gradients trained locally by the clients
- Other tasks that the server agent needs to perform to manage the federated learning process (e.g., recording the sample size of each client)

Specifically, the current server agent has the following functionalities.

.. note::

    Users can also define their own functionalities, either by inheriting from the ``ServerAgent`` class or by adding new methods directly to the current server agent. If you think your added functionalities would be useful to other users, please consider contributing them to APPFL by submitting a pull request.

.. code-block:: python

    class ServerAgent:
        def __init__(
            self,
            server_agent_config: ServerAgentConfig = ServerAgentConfig()
        ) -> None:
            """
            Initialize the server agent with the configurations.
            """

        def get_client_configs(self, **kwargs) -> DictConfig:
            """
            Return the FL configurations that are shared among all clients.
            """

        def global_update(
            self,
            client_id: Union[int, str],
            local_model: Union[Dict, OrderedDict, bytes],
            blocking: bool = False,
            **kwargs
        ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Update the global model using the local model from a client and return the updated global model.
            :param: `client_id`: A unique client id for the server to distinguish clients, which can be obtained via `ClientAgent.get_id()`.
            :param: `local_model`: The local model from a client, can be serialized bytes.
            :param: `blocking`: The global model may not be immediately available for certain aggregation methods (e.g., any synchronous method).
                Setting `blocking` to `True` will block the client until the global model is available.
                Otherwise, the method may return a `Future` object if the most up-to-date global model is not yet available.
            :return: The updated global model (as a Dict or OrderedDict), and optional metadata (as a Dict) if `blocking` is `True`.
                Otherwise, return the `Future` object of the updated global model and optional metadata.
            """

        def get_parameters(
            self,
            blocking: bool = False,
            **kwargs
        ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Return the global model to the clients.
            :param: `blocking`: The global model may not be immediately available (e.g., if the server wants to wait for all clients
                to send the `get_parameters` request before returning the global model so that all clients start from the same initial model).
                Setting `blocking` to `True` will block the client until the global model is available.
            :param: `kwargs`: Additional arguments for the method. Specifically,
                - `init_model`: whether to get the initial model (which should be the same for all clients, thus blocking)
                - `serial_run`: set to `True` for serial simulation runs, in which no blocking is needed.
                - `globus_compute_run`: set to `True` for Globus Compute runs, in which no blocking is needed.
            """

        def set_sample_size(
                self,
                client_id: Union[int, str],
                sample_size: int,
                sync: bool = False,
                blocking: bool = False,
            ) -> Optional[Union[Dict, Future]]:
            """
            Set the size of the local dataset of a client.
            :param: client_id: A unique client id for the server to distinguish clients, which can be obtained via `ClientAgent.get_id()`.
            :param: sample_size: The size of the local dataset of a client.
            :param: sync: Whether to synchronize the sample size among all clients. If `True`, the method can return the relative weight of the client.
            :param: blocking: Whether to block the client until the sample size of all clients is synchronized.
                If `True`, the method will return the relative weight of the client.
                Otherwise, the method may return a `Future` object of the relative weight, which will be resolved
                when the sample size of all clients is synchronized.
            """

        def training_finished(self, internal_check: bool = False) -> bool:
            """
            Indicate whether the training is finished.
            :param: `internal_check`: Whether this is an internal check (e.g., for the server to check if the training is finished) or it is requested by the client.
            """

        def server_terminated(self):
            """
            Indicate whether the server can stop listening to the clients.
            """

Configurations
--------------

As shown above, creating a server agent requires a server agent configuration, defined by the ``appfl.config.ServerAgentConfig`` class, which can be loaded directly from a YAML file. An example server agent configuration YAML file is shown below.

The configuration file is composed of two main parts:

- ``client_configs``: The configurations shared among all clients. This part defines the configurations for the federated learning process, including the trainer type, the model architecture, the compression configurations, etc.
- ``server_configs``: The configurations for the server agent, including the aggregation method, the scheduling method, etc.


.. literalinclude:: ../_static/server_fedavg.yaml
    :language: yaml
    :caption: Server agent configuration YAML file for FedAvg algorithm on the MNIST dataset with a simple CNN model.

Client Configurations
~~~~~~~~~~~~~~~~~~~~~

The client configurations shared among all clients are composed of three main components:

- ``train_configs``: This component contains all training-related configurations, which can be further divided into the following sub-components:

    - *Trainer configurations*: Note that the required trainer configurations depend on the trainer you use. You can also define your own trainer with any additional configurations you need, and then provide those configurations under ``client_configs.train_configs`` in the server configuration YAML file.

        - ``trainer``: The class name of the trainer to use for client local training. The trainer should be defined in ``src/appfl/algorithm/trainer``. For example, ``VanillaTrainer`` simply updates the model for a certain number of epochs or batches.
        - ``mode``: For ``VanillaTrainer``, ``mode`` is a required configuration whose allowed values are ``epoch`` and ``step``, specifying whether to train for a certain number of epochs or a certain number of steps (batches).
        - ``num_local_steps`` / ``num_local_epochs``: Number of steps (if ``mode=step``) or epochs (if ``mode=epoch``) for an FL client in each local training round.
        - ``optim``: Name of the optimizer to use from the ``torch.optim`` module.
        - ``optim_args``: Keyword arguments for the selected optimizer.
        - ``do_validation``: Whether to perform client-side validation in each training round.
        - ``do_pre_validation``: Whether to perform client-side validation prior to local training.
        - ``use_dp``: Whether to use differential privacy.
        - ``epsilon``, ``clip_grad``, ``clip_value``, ``clip_norm``: Parameters used if differential privacy is enabled.
    - *Loss function*: To specify the loss function to use during local training, we provide two options:

        - Loss function from ``torch``: Provide the name of a loss function available in ``torch.nn`` (e.g., ``CrossEntropyLoss``) in ``loss_fn`` and its arguments in ``loss_fn_kwargs``.
        - Loss function defined in a local file: Define your own loss function by inheriting from ``nn.Module`` and implementing its ``forward()`` method. Then provide the path to the loss function file in ``loss_fn_path`` and the class name of the loss function in ``loss_fn_name``.
    - *Metric function*: To specify the metric function used during validation, provide the path to the file containing the metric function in ``metric_path`` and the name of the metric function in ``metric_name``.
    - *Dataloader settings*: While the server-side configuration does not contain any information about each client's local dataset, it can specify how the dataset is converted to a dataloader, such as the batch size and whether to shuffle.
- ``model_configs``: This component contains the definition of the machine learning model used in the FL experiment. The model architecture should be defined as a ``torch.nn.Module`` in a local file on the server side, and the following information should be provided:

    - ``model_path``: Path to the model definition file.
    - ``model_name``: Class name of the defined model.
    - ``model_kwargs``: Keyword arguments for initializing the model.
- ``comm_configs``: This component contains the settings for the communication between the FL server and clients, such as the ``compressor_configs``.

Server Configurations
~~~~~~~~~~~~~~~~~~~~~

The server configurations contain the following key components:

- *Scheduler configurations*: The name of the scheduler (``scheduler``) and its keyword arguments (``scheduler_kwargs``). All supported schedulers are available in ``src/appfl/algorithm/scheduler``.
- *Aggregator configurations*: The name of the aggregator (``aggregator``) and its keyword arguments (``aggregator_kwargs``). All supported aggregators are available in ``src/appfl/algorithm/aggregator``.
- *Communicator configurations*: The configurations for the communication between the server and clients, such as the ``grpc_configs``.
- *Logging configurations and others*: The configurations for logging, such as ``logging_output_dirname`` and ``logging_output_filename``, as well as ``num_global_epochs``.

.. note::

    You may notice that both ``server_configs`` and ``client_configs`` have a ``comm_configs`` field. When the server agent is created, its communication configurations are the merge of ``server_configs.comm_configs`` and ``client_configs.comm_configs``. However, ``client_configs.comm_configs`` is also shared with the clients, while ``server_configs.comm_configs`` is not. As we want the clients to be aware of the compressor configurations, we put ``compressor_configs`` under ``client_configs.comm_configs`` to share with the clients during the FL experiment.
