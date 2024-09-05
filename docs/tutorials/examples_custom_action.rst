Example: Add a Custom Action
============================

In ``APPFL``, the server supports several actions such as getting general client configurations, getting the global model parameters, and updating the global model parameters (i.e., federated training). However, in some cases, you may want to add a custom action to the server, such as fedearated evaluation. In this example, we show how to add a custom action to the server to generate a data readiness report for all the clients local datasets.    

Client Side Implementation
--------------------------

In this example, we focus on the client-driven communication pattern (MPI and gRPC), where the clients sends requests to the server for any actions they want to perform. In this case, the client-side can simply define a function to generate the data readiness report for its local dataset, and then send a request to the server to generate the report for all the clients. 

However, as the ``APPFL`` defines client agent ``appfl.agent.ClientAgent`` to act on behalf of the client, we highly recommend to define the custom action within the client agent either by extending the ``appfl.agent.ClientAgent`` or by adding a new method to the existing ``appfl.agent.ClientAgent``. In this example, we create a new client agent by extending the existing client agent and adding a new method to generate the data readiness report. 

.. note::
    If you think your custom action is useful for the community, please consider define it within the ``appfl.client.ClientAgent`` directly and contribute it to the ``APPFL`` framework by creating a pull request.


New Client Agent
~~~~~~~~~~~~~~~~

In this example, we define a `DRAgent` as below which contains a simple function, ``generate_mnist_readiness_report``, to generate some data readiness metrics for the MNIST dataset.

.. literalinclude:: ./examples/dr_metric/appfl_dr_metric_grpc/client/resources/dr_agent.py
    :language: python
    :caption: DRAgent - A simple client agent which can generate data readiness metrics for the MNIST dataset.

Send Request to the Server
~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether it is MPI or gRPC communicator, ``APPFL`` provides an interface function, ``invoke_custom_action``, for user to send any custom request to the server, as shown below. The handler for the custom action should be defined on the server side, as shown in the next section.

.. note::

    See the sections "Launch Experiments" for details on how to create the communicator.


.. code-block:: python

    client_communicator = ... # this can be either MPI or gRPC communicator, see the following sections
    data_readiness = client_agent.generate_mnist_readiness_report()
    client_communicator.invoke_custom_action(action='get_mnist_readiness_report', **data_readiness)


Server Side Implementation
--------------------------

On the server side, the server should update a handler for processing the custom action request from the client. In this example, we will show how to update the custom action handler for both the MPI and gRPC communicator.

MPI Communicator
~~~~~~~~~~~~~~~~

User needs to update the ``APPFL`` source code's MPI server communicator at ``appfl.comm.mpi.mpi_server_communicator.MPIServerCommunicator``. You need to update its ``_invoke_custom_action`` method to handle the custom action request from the client. Here is an example implementation for action ``get_mnist_readiness_report`` which simply prints the readiness report for all the clients.

.. code-block:: python

    def _invoke_custom_action(
        self,
        client_id: int,
        request: MPITaskRequest,
    ) -> Optional[MPITaskResponse]:
        ...
        if action == "set_sample_size":
            ...
        elif action == "close_connection":
            ...
        elif action == "get_mnist_readiness_report":
            # A very simple example of readiness report generation
            num_clients = self.server_agent.get_num_clients()
            if not hasattr(self, "_dr_metrics_lock"):
                self._dr_metrics_lock = threading.Lock()
                self._dr_metrics_req_count = 0
                self._dr_metrics = {}
            with self._dr_metrics_lock:
                self._dr_metrics_req_count += 1
                for k, v in meta_data.items():
                    if k not in self._dr_metrics:
                        self._dr_metrics[k] = {}
                    self._dr_metrics[k][client_id] = v
                if self._dr_metrics_req_count == num_clients:
                    print("Printing readiness report...")
                    print(self._dr_metrics)
            return MPITaskResponse(status=MPIServerStatus.RUN.value)
        else:
            raise NotImplementedError(f"Custom action {action} is not implemented.")

gRPC Communicator
~~~~~~~~~~~~~~~~~

Similar to the MPI communicator, user needs to update the ``APPFL`` source code's gRPC server communicator at ``appfl.comm.grpc.grpc_server_communicator.GRPCServerCommunicator``. You need to update its ``InvokeCustomAction`` method to handle the custom action request from the client. Here is an example implementation for action ``get_mnist_readiness_report`` which simply prints the readiness report for all the clients.

.. code-block:: python

    def InvokeCustomAction(self, request, context):
        action = request.action
        meta_data = json.loads(request.meta_data) if len(request.meta_data) > 0 else {}
        if action == "set_sample_size":
            ...
        elif action == "close_connection":
            ...
        elif action == "get_mnist_readiness_report":
            # A very simple example of readiness report generation
            num_clients = self.server_agent.get_num_clients()
            if not hasattr(self, "_dr_metrics_lock"):
                self._dr_metrics_lock = threading.Lock()
                self._dr_metrics_req_count = 0
                self._dr_metrics = {}
            with self._dr_metrics_lock:
                self._dr_metrics_req_count += 1
                for k, v in meta_data.items():
                    if k not in self._dr_metrics:
                        self._dr_metrics[k] = {}
                    self._dr_metrics[k][client_id] = v
                if self._dr_metrics_req_count == num_clients:
                    print("Printing readiness report...")
                    print(self._dr_metrics)
            return CustomActionResponse(header=ServerHeader(status=ServerStatus.RUN))
        else:
            raise NotImplementedError(f"Custom action {action} is not implemented.")
            
Launch Experiments
------------------

The following subsections show how to launch the experiments using the MPI and gRPC communicator.

MPI Communicator
~~~~~~~~~~~~~~~~

The source code for this example experiment is located at ``docs/tutorials/examples/dr_metric/appfl_dr_metric_mpi``, and the files are organized as follows:

.. code-block:: bash

    appfl_dr_metric_mpi
    ├── resources
    │   ├── __init__.py
    │   ├── mnist_dataset.py    # MNIST dataset loader
    │   ├── cnn.py              # CNN model for MNIST dataset
    │   ├── acc.py              # evaluation metric for the CNN model on MNIST dataset
    │   └── dr_agent.py         # client agent for generating data readiness report
    ├── config_client.yaml      # client configuration file
    ├── config_server.yaml      # server configuration file
    └── run_mpi.py              # script to run the experiment

You can run the experiment using the following command:

.. code-block:: bash

    mpiexec -n 6 python run_mpi.py

.. note::

    The above command will run the experiment with 5 clients: one MPI process for the server and 5 MPI processes for the clients.

gRPC Communicator
~~~~~~~~~~~~~~~~~

We provide a similar example for the gRPC communicator at ``docs/tutorials/examples/dr_metric/appfl_dr_metric_grpc``. The files are organized as follows:

.. code-block:: bash

    appfl_dr_metric_grpc
    ├── client
    │   ├── resources
    │   │   ├── __init__.py
    │   │   ├── mnist_dataset.py    # MNIST dataset loader
    │   │   └── dr_agent.py         # client agent for generating data readiness report
    │   ├── client1_conifg.yaml     # configuration file for client 1
    │   ├── client2_conifg.yaml     # configuration file for client 2
    │   └── run_client.py           # client script to run the experiment
    └── server
        ├── resources
        │   ├── cnn.py              # CNN model for MNIST dataset
        │   └── acc.py              # evaluation metric for the CNN model on MNIST dataset
        ├── config.yaml             # server configuration file
        └── run_server.py           # server script to run the experiment

To run the experiment, you need to start the server first in one terminal, and then start two clients in two separate terminals. You can run the server using the following command:

.. code-block:: bash

    cd docs/tutorials/examples/dr_metric/appfl_dr_metric_grpc/server
    python run_server.py

You can run the clients using the following commands:

.. code-block:: bash

    cd docs/tutorials/examples/dr_metric/appfl_dr_metric_grpc/client
    python run_client.py --config client1_config.yaml
    python run_client.py --config client2_config.yaml       # run this in a separate terminal

You will see the readiness report printed on the server terminal once all the clients have sent their readiness metrics, as shown below:

.. code-block:: bash

    Printing readiness report...
    {'ci': {'10854bb8-6042-4867-85bc-bf61a6810b7f': 0.1135829860901944, '4b6281af-f979-475f-b658-571d27624d23': 0.1805402082160045}, 'ss': {'10854bb8-6042-4867-85bc-bf61a6810b7f': 35861, '4b6281af-f979-475f-b658-571d27624d23': 24139}}