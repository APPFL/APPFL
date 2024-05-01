APPFL Communicator
==================

The APPFL communicator is used for exchanging 

- various types of model data (e.g. parameters, gradients, compressed bytes, etc.)
- metadata such as configurations, control signals, etc.

for different tasks on the server/client agent side to run.

In APPFL, we support the following types of communication protocols:

- :ref:`MPI: Message Passing Interface`
- :ref:`gRPC: Google Remote Procedure Call`
- :ref:`Globus Compute`

MPI: Message Passing Interface
------------------------------

MPI can be used for simulating federated learning on a single machine or a cluster of machines. It is composed of two parts:

- MPI Server Communicator (`appfl.comm.mpi.MPIServerCommunicator`) which starts a server to listen to incoming requests from clients for various tasks.
- MPI Client Communicator (`appfl.comm.mpi.MPIClientCommunicator`) which sends requests to the server for various tasks.

MPI Server Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

For the server side, the server only needs to create an instance of `MPIServerCommunicator` and call the `serve` method to start the server. The server will listen to incoming requests from clients for various tasks.

The server can handle the following tasks:

- Get configurations that are shared among all clients via the `_get_configuration` method.
- Get the global model via the `_get_global_model` method.
- Update the global model with the local model from the client via the `_update_global_model` method.
- Invoke custom action on the server via the `_invoke_custom_action` method.

.. note::
    
    The server will automatically stop itself after reaching the specified `num_global_epochs`.

.. note::
    
    You can add any custom tasks by implementing the corresponding methods in the `_invoke_custom_action` class.

.. code:: python

    class MPIServerCommunicator:
        def __init__(
            self, 
            comm,
            server_agent: APPFLServerAgent,
            logger: Optional[ServerAgentFileLogger] = None,
        ) -> None:
            """
            Create an MPI server communicator.
            :param `comm`: MPI communicator object
            :param `server_agent`: `APPFLServerAgent` object
            :param `logger`: A logger object for logging messages
            """

        def serve(self):
            """
            Start the server to serve the clients.
            """
            
        def _get_configuration(
            self, 
            client_id: int, 
            request: MPITaskRequest
        ) -> MPITaskResponse:
            """
            Client requests the FL configurations that are shared among all clients from the server.
            :param: `client_id`: A unique client ID (only for logging purpose now)
            :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
            :return `response.status`: Server status
            :return `response.meta_data`: JSON serialized FL configurations
            """
        
        def _get_global_model(
            self, 
            client_id: int, 
            request: MPITaskRequest
        ) -> Optional[MPITaskResponse]:
            """
            Return the global model to clients. This method is supposed to be called by 
                clients to get the initial and final global model.
            :param: `client_id`: A unique client ID, which is the rank of the client in 
                MPI (only for logging purpose now)
            :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
            :return `response.status`: Server status
            :return `response.payload`: Serialized global model
            :return `response.meta_data`: JSON serialized metadata dictionary (if needed)
            """

        def _update_global_model(
            self, 
            client_id: int, 
            request: MPITaskRequest
        ) -> Optional[MPITaskResponse]:
            """
            Update the global model with the local model from the client, 
            and return the updated global model to the client.
            :param: `client_id`: A unique client ID, which is the rank of the client in MPI.
            :param: `request.payload`: Serialized local model
            :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
            :return `response.status`: Server status
            :return `response.payload`: Serialized updated global model
            :return `response.meta_data`: JSON serialized metadata dictionary (if needed)
            """

        def _invoke_custom_action(
            self,
            client_id: int,
            request: MPITaskRequest,
        ) -> Optional[MPITaskResponse]:
            """
            Invoke custom action on the server.
            :param: `client_id`: A unique client ID, which is the rank of the client in
                MPI (only for logging purpose now)
            :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
            :return `response.status`: Server status
            :return `response.meta_data`: JSON serialized metadata dictionary (if needed)
            """

MPI Client Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

During the federated learning process, the client can communicates to the server by invoking corresponding methods in the `MPIClientCommunicator` class. For example, after a client finish a local training round, it can send the local model to the server for global aggregation by calling the `update_global_model` method.

.. note:: 

    You can add any custom tasks by implementing the corresponding methods in the `invoke_custom_action` class. Also make sure that the server has the corresponding handler codes implemented in the `_invoke_custom_action` method.

.. code:: python

    class MPIClientCommunicator:
        def __init__(
            self,
            comm,
            server_rank: int,
        ):
            """
            Create an MPI client communicator.
            :param `comm`: MPI communicator object
            :param `server_rank`: Rank of the MPI process that is running the server
            """

        def get_configuration(self, **kwargs) -> DictConfig:
            """
            Get the federated learning configurations from the server.
            :param kwargs: additional metadata to be sent to the server
            :return: the federated learning configurations
            """
        
        def get_global_model(self, **kwargs) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Get the global model from the server.
            :param kwargs: additional metadata to be sent to the server
            :return: the global model with additional metadata (if any)
            """
            
        def update_global_model(self, local_model: Union[Dict, OrderedDict, bytes], **kwargs) -> Tuple[Union[Dict, OrderedDict], Dict]:
            """
            Send local model to FL server for global update, and return the new global model.
            :param local_model: the local model to be sent to the server for gloabl aggregation
            :param kwargs: additional metadata to be sent to the server
            :return: the updated global model with additional metadata. Specifically, `meta_data["status"]` is either "RUNNING" or "DONE".
            """

        def invoke_custom_action(self, action: str, **kwargs) -> Dict:
            """
            Invoke a custom action on the server.
            :param action: the action to be invoked
            :param kwargs: additional metadata to be sent to the server
            :return: the response from the server (if any)
            """

Example Usage
~~~~~~~~~~~~~

Here is an example of how to use the MPI communicator in APPFL to start FL experiments.


.. literalinclude:: ../../examples/mpi/run_mpi.py
    :language: python
    :caption: Running Federated Learning with MPI Communicator.

gRPC: Google Remote Procedure Call
----------------------------------

gRPC can be used either for simulating federated learning on a single machine or cluster, or for running federated learning on real-world distributed machines. It is composed of two parts:

- gRPC Server Communicator (`appfl.comm.grpc.GRPCServerCommunicator`) which creats a server for listenning to incoming requests from clients for various tasks.
- gRPC Client Communicator (`appfl.comm.grpc.GRPCClientCommunicator`) which sends requests to the server for various tasks.

gRPC Server Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

For the server side, the server only needs to create an instance of `GRPCServerCommunicator` and call the `serve` method (available in `appfl.comm.grpc`) to start the server. The server will listen to incoming requests from clients for various tasks.

The server can handle the following tasks:

- Get configurations that are shared among all clients via the `GetConfiguration` method.
- Get the global model via the `GetGlobalModel` method.
- Update the global model with the local model from the client via the `UpdateGlobalModel` method.
- Invoke custom action on the server via the `InvokeCustomAction` method.

.. note:: 

    You can add any custom tasks by implementing the corresponding methods in the `InvokeCustomAction` class.

.. code:: python

    class GRPCServerCommunicator(GRPCCommunicatorServicer):
        def __init__(
            self,
            server_agent: APPFLServerAgent,
            max_message_size: int = 2 * 1024 * 1024,
            logger: Optional[ServerAgentFileLogger] = None,
        ) -> None:
            """
            Creates a gRPC server communicator.
            :param `server_agent`: `APPFLServerAgent` object
            :param `max_message_size`: Maximum message size in bytes to be sent/received. 
                Object size larger than this will be split into multiple messages.
            :param `logger`: A logger object for logging messages
            """

        def GetConfiguration(self, request, context):
            """
            Client requests the FL configurations that are shared among all clients from the server.
            :param: `request.header.client_id`: A unique client ID
            :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
            :return `response.header.status`: Server status
            :return `response.configuration`: JSON serialized FL configurations
            """
        
        def GetGlobalModel(self, request, context):
            """
            Return the global model to clients. This method is supposed to be called by 
            clients to get the initial and final global model. Returns are sent back as a 
            stream of messages.
            :param: `request.header.client_id`: A unique client ID
            :param: `request.meta_data`: JSON serialized metadata dictionary (if needed)
            :return `response.header.status`: Server status
            :return `response.global_model`: Serialized global model
            """

        def UpdateGlobalModel(self, request_iterator, context):
            """
            Update the global model with the local model from a client. This method will 
            return the updated global model to the client as a stream of messages.
            :param: request_iterator: A stream of `DataBuffer` messages - which contains 
                serialized request in `UpdateGlobalModelRequest` type.

            If concatenating all messages in `request_iterator` to get a `request`, then
            :param: request.header.client_id: A unique client ID
            :param: request.local_model: Serialized local model
            :param: request.meta_data: JSON serialized metadata dictionary (if needed)
            """

        def InvokeCustomAction(self, request, context):
            """
            This function is the entry point for any custom action that the server agent 
            can perform. The server agent should implement the custom action and call this
            function to perform the action.
            :param: `request.header.client_id`: A unique client ID
            :param: `request.action`: A string tag representing the custom action
            :param: `request.meta_data`: JSON serialized metadata dictionary for the custom action (if needed)
            :return `response.header.status`: Server status
            :return `response.meta_data`: JSON serialized metadata dictionary for return values (if needed)
            """

gRPC Client Communicator
~~~~~~~~~~~~~~~~~~~~~~~~

During the federated learning process, the client can communicate to the server by invoking corresponding methods in the `GRPCClientCommunicator` class. For example, after a client finish a local training round, it can send the local model to the server for global aggregation by calling the `update_global_model` method.

.. note:: 

    You can add any custom tasks by implementing the corresponding methods in the `invoke_custom_action` class. Also make sure that the server has the corresponding handler codes implemented in the `InvokeCustomAction` method.

.. code:: python

    class GRPCClientCommunicator:
        def __init__(
            self, 
            client_id: Union[str, int],
            *,
            server_uri: str,
            use_ssl: bool = False,
            use_authenticator: bool = False,
            root_certificate: Optional[Union[str, bytes]] = None,
            authenticator: Optional[str] = None,
            authenticator_args: Dict[str, Any] = {},
            max_message_size: int = 2 * 1024 * 1024,
            **kwargs,
        ):
            """
            Create a channel to the server and initialize the gRPC client stub.
            
            :param client_id: A unique client ID.
            :param server_uri: The URI of the server to connect to.
            :param use_ssl: Whether to use SSL/TLS to authenticate the server and encrypt communicated data.
            :param use_authenticator: Whether to use an authenticator to authenticate the client in each RPC. Must have `use_ssl=True` if `True`.
            :param root_certificate: The PEM-encoded root certificates as a byte string, or `None` to retrieve them from a default location chosen by gRPC runtime.
            :param authenticator: The name of the authenticator to use for authenticating the client in each RPC.
            :param authenticator_args: The arguments to pass to the authenticator.
            :param max_message_size: The maximum message size in bytes.
            """

        def get_configuration(self, **kwargs) -> DictConfig:
            """
            Get the federated learning configurations from the server.
            :param kwargs: additional metadata to be sent to the server
            :return: the federated learning configurations
            """
            
        def get_global_model(self, **kwargs) -> Union[Union[Dict, OrderedDict], Tuple[Union[Dict, OrderedDict], Dict]]:
            """
            Get the global model from the server.
            :param kwargs: additional metadata to be sent to the server
            :return: the global model with additional metadata (if any)
            """

        def update_global_model(self, local_model: Union[Dict, OrderedDict, bytes], **kwargs) -> Tuple[Union[Dict, OrderedDict], Dict]:
            """
            Send local model to FL server for global update, and return the new global model.
            :param local_model: the local model to be sent to the server for gloabl aggregation
            :param kwargs: additional metadata to be sent to the server
            :return: the updated global model with additional metadata. Specifically, `meta_data["status"]` is either "RUNNING" or "DONE".
            """
            
        def invoke_custom_action(self, action: str, **kwargs) -> Dict:
            """
            Invoke a custom action on the server.
            :param action: the action to be invoked
            :param kwargs: additional metadata to be sent to the server
            :return: the response from the server
            """

Example Usage
~~~~~~~~~~~~~

Below shows an example on how to start a server using `GRPCServerCommunicator`, which waits for incoming requests from clients. 

.. literalinclude:: ../../examples/grpc/run_server.py
    :language: python
    :caption: Running Federated Learning with gRPC Server Communicator.

To interact with the server and start an FL experiment, you can start a client using `GRPCClientCommunicator` as shown below.

.. literalinclude:: ../../examples/grpc/run_client_1.py
    :language: python
    :caption: Running Federated Learning with gRPC Client Communicator.

Globus Compute
--------------

TBA