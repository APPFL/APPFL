APPFL Client Agent
==================

Functionalities
---------------

The APPFL client agent acts on behalf of the FL client to perform various tasks related to local training, including:

- Load general configurations shared among all clients from the server.
- Load model parameters received from the server.
- Perform the local training job according to the configuration.
- Prepare local model parameters and other optional metadata for communication with the server.
- Other tasks that the client agent needs to perform for certain FL algorithms (e.g., getting the size of the local dataset).

Specifically, the current client agent has the following functionalities.

.. note::

    Users can also define their own functionalities, either by inheriting from the ``ClientAgent`` class or by adding new methods directly to the current client agent. If you think your added functionalities would be useful to other users, please consider contributing them to APPFL by submitting a pull request.

.. code-block:: python

    class ClientAgent:
        def __init__(
            self,
            client_agent_config: ClientAgentConfig = ClientAgentConfig()
        ) -> None:
            """
            Initialize the client agent with configurations.
            """

        def load_config(self, config: DictConfig) -> None:
            """
            Load additional configurations provided by the server.
            """

        def get_id(self) -> str:
            """
            Return a unique client id for the server to distinguish clients.
            """

        def get_sample_size(self) -> int:
            """
            Return the size of the local dataset.
            """

        def train(self) -> None:
            """
            Train the model locally.
            """

        def get_parameters(self) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
            """
            Return parameters for communication.
            :return parameters: The parameters to be sent to the server,
                can be of type Dict, OrderedDict, bytes (if compressed), or
                Tuple[Union[Dict, OrderedDict, bytes], Dict] with optional
                metadata in Dict type.
            """

        def load_parameters(self, params) -> None:
            """
            Load parameters from the server.
            """

Configurations
--------------

As shown above, creating a client agent requires a client agent configuration, defined by the ``appfl.config.ClientAgentConfig`` class, which can be loaded from a YAML file. The following is an example client agent configuration YAML file.

.. note::

    The client YAML file only contains configurations that are **specific** to one client. The general configurations shared among all clients are sent from the server.

.. literalinclude:: ../_static/client.yaml
    :language: yaml
    :caption: Client agent configuration YAML file for a specific client.


The above configuration file contains several client-**specific** configurations, such as the device to use and how to load the private local dataset:

- ``client_id``: A unique identifier for the client among all clients in the current FL experiment. The server uses this identifier to distinguish clients and for logging purposes.
- ``data_configs``: The most important part of the client configuration file. It contains the path on the client's local machine to the file that defines how to load the local dataset (``dataset_path``), the name of the dataset-loading function in that file (``dataset_name``), and any keyword arguments it needs (``dataset_kwargs``).
- ``train_configs``: The device to use for training and some logging configurations.
- ``comm_configs``: Communication settings the client may need to connect to the server. For example, if the experiment uses gRPC as the communication method, then the client needs to specify the ``server_uri``, ``max_message_size``, and ``use_ssl`` to establish the connection to the server.
