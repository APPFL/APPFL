Running with gRPC
=================

gRPC allows clients from different platforms to seamlessly connect to the server for federated learning.
This contrasts with MPI where all clients and servers should reside in the same cluster.

gRPC uses the HTTP/2 protocol.
A server hosts a service specified by a URI (e.g., ``moonshot.cels.anl.gov:50051`` where ``50051`` is the port number) for communication and clients send requests and receive responses via that URI. Communication protocols between a server and clients are defined via `Protocol Buffers <https://developers.google.com/protocol-buffers/docs/overview>`_, which are defined in the ``appfl/protos`` directory.
For more details, we refer to `gRPC <https://grpc.io/docs/>`_.

To use gRPC, we first need to launch a server as follows:

.. code-block:: console

    $ python appfl/run_grpc_server.py

It will read the configuration file ``appfl/config/config.yaml`` to identify the number of clients and its service URI.
For example, the following specifies that the number of clients to connect is ``1`` and the URI is ``moonshot.cels.anl.gov:50051``.

.. code-block:: yaml

    num_clients: 1
    server:
        id: 1
        host: moonshot.cels.anl.gov
        port: 50051

Once it is launched, it will listen to the port to serve requests from clients.

For a client to connect to the server, you need to run the following:

.. code-block:: console

    $ python appfl/run_grpc_client.py

It will try to connect to the server specified in the ``config.yaml`` file.
We note that the server will not start training until the specified number of clients connect to the server.

Below is an example of running gPRC.

.. code-block:: console
    :caption: Log messages from the server
    :linenos:

    $ python appfl/run_grpc_server.py
    Starting the server to listen to requests from clients . . .
    [2021-11-12 14:57:56,149][protos.server][INFO] - Received JobRequest from client 1 job_done 0
    [2021-11-12 14:57:56,152][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv1.weight,1)
    [2021-11-12 14:57:56,155][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv1.bias,1)
    [2021-11-12 14:57:56,157][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv2.weight,1)
    [2021-11-12 14:57:56,161][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv2.bias,1)
    [2021-11-12 14:57:56,162][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc1.weight,1)
    [2021-11-12 14:57:56,178][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc1.bias,1)
    [2021-11-12 14:57:56,180][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc2.weight,1)
    [2021-11-12 14:57:56,181][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc2.bias,1)
    [2021-11-12 14:59:28,391][protos.operator][INFO] - [Round:  001] Finished; all clients have sent their results.
    [2021-11-12 14:59:28,392][protos.operator][INFO] - [Round:  001] Updating model weights
    [2021-11-12 14:59:30,909][protos.operator][INFO] - [Round:  001] Test set: Average loss: 0.0238, Accuracy: 99.27%
    [2021-11-12 14:59:30,913][protos.server][INFO] - Received JobRequest from client 1 job_done 2
    [2021-11-12 14:59:30,915][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv1.weight,2)
    [2021-11-12 14:59:30,916][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv1.bias,2)
    [2021-11-12 14:59:30,917][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv2.weight,2)
    [2021-11-12 14:59:30,918][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,conv2.bias,2)
    [2021-11-12 14:59:30,921][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc1.weight,2)
    [2021-11-12 14:59:30,937][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc1.bias,2)
    [2021-11-12 14:59:30,938][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc2.weight,2)
    [2021-11-12 14:59:30,939][protos.server][INFO] - Received TensorRequest from (client,name,round)=(1,fc2.bias,2)
    [2021-11-12 15:01:03,231][protos.operator][INFO] - [Round:  002] Finished; all clients have sent their results.
    [2021-11-12 15:01:03,232][protos.operator][INFO] - [Round:  002] Updating model weights
    [2021-11-12 15:01:05,639][protos.operator][INFO] - [Round:  002] Test set: Average loss: 0.0236, Accuracy: 99.22%

.. code-block:: console
    :caption: Log messages from a client
    :linenos:
    :lineno-start: 27

    $ python appfl/run_grpc_client.py
    [2021-11-12 14:57:56,151][protos.client][INFO] - Received JobReponse with (server,round,job)=(1,1,2)
    [2021-11-12 14:57:56,184][algorithm.fedavg][INFO] - [Client ID:  01, Local epoch:  001]
    [2021-11-12 14:58:28,284][algorithm.fedavg][INFO] - [Client ID:  01, Local epoch:  002]
    [2021-11-12 14:58:57,890][algorithm.fedavg][INFO] - [Client ID:  01, Local epoch:  003]
    [2021-11-12 14:59:30,911][__main__][INFO] - [Client ID:  01] Trained and sent results back to the server
    [2021-11-12 14:59:30,914][protos.client][INFO] - Received JobReponse with (server,round,job)=(1,2,2)
    [2021-11-12 14:59:30,942][algorithm.fedavg][INFO] - [Client ID:  01, Local epoch:  001]
    [2021-11-12 15:00:02,341][algorithm.fedavg][INFO] - [Client ID:  01, Local epoch:  002]
    [2021-11-12 15:00:32,673][algorithm.fedavg][INFO] - [Client ID:  01, Local epoch:  003]
    [2021-11-12 15:01:05,640][__main__][INFO] - [Client ID:  01] Trained and sent results back to the server
    [2021-11-12 15:01:05,643][protos.client][INFO] - Received JobReponse with (server,round,job)=(1,2,3)

We briefly describe the log messages:

#. Once all clients connect to the server, they request information about the job to perform (line 3).
#. Perform each round of training until termination conditions are met:

    #. Clients request information about tensors constituting the model to train to the server (lines 4--11).
    #. Clients start training (lines 29--31).
    #. Clients finish local training and send the resulting tensors back to the server (line 32)
    #. Server aggregates the tensors from clients and determines if termination conditions are met (lines 12--14).


