Defining your server agent
==========================

.. note::

    We always welcome your contributions to ``APPFL`` via pull requests.

When your server agent needs additional functionalities that are not provided by the current ``ServerAgent``, you can create your own server agent by inheriting from ``appfl.agent.ServerAgent``. To make the new functionalities available to the clients, modify the ``appfl.comm.mpi.MPIServerCommunicator._invoke_custom_action`` method (for MPI) or ``appfl.comm.grpc.GRPCServerCommunicator.InvokeCustomAction`` (for gRPC) to call your new functionalities under a corresponding action name.

For example, suppose you create a new server agent with an additional functionality for federated inference, as below:

.. code-block:: python

    class MyServerAgentWithFedInference(ServerAgent):
        def fl_inference(*args, **kwargs):
            pass

You can then make this new function available to the clients by calling it in the custom action method of the corresponding communicator.

.. code-block:: python


    # For gRPC communicator
    class GRPCServerCommunicator(GRPCCommunicatorServicer):
        ...
        def InvokeCustomAction(self, request, context):
            ...
            if action == 'set_sample_size':
                ...
            elif action == 'fl_inference':
                ...
                val = self.server_agent.fl_inference()
                ...
            ...

    # For MPI communicator
    class MPIServerCommunicator:
        ...
        def _invoke_custom_action(
            self,
            client_id: int,
            request: MPITaskRequest,
        ) -> Optional[MPITaskResponse]:
            ...
            if action == "set_sample_size":
                ...
            elif action == 'fl_inference':
                ...
                val = self.server_agent.fl_inference()
                ...
            ...
