Running with gRPC
=================

gRPC allows clients from different platforms to seamlessly connect to the server for federated learning.
This contrasts with MPI where all clients and servers should reside in the same cluster.

gRPC uses the HTTP/2 protocol.
A server hosts a service specified by a URI (e.g., ``moonshot.cels.anl.gov:50051`` where ``50051`` is the port number) for communication and clients send requests and receive responses via that URI. Communication protocols between a server and clients are defined via `Protocol Buffers <https://developers.google.com/protocol-buffers/docs/overview>`_, which are defined in the ``appfl/protos`` directory.
For more details, we refer to `gRPC <https://grpc.io/docs/>`_.

The APIs to run gRPC are defined in ``appfl/run_grpc_server.py`` and ``appfl/run_grpc_client.py`` as functions ``run_server()`` and ``run_client()`` for server and client, respectively.
The following function defined in ``run_grpc_server.py`` shows how to launch a server:

.. code-block:: python

    run_server(cfg, comm_rank, model, test_dataset, num_clients, DataSet_name)

where ``cfg`` represents the ``DictConfig`` object that contains the configuration file ``appfl/config/config.yaml``, ``comm_rank`` the rank of the server, ``model`` the neural network model that will be learned via federated learning, ``test_dataset`` test data to check the accuracy of the ``model``, ``num_clients`` the number of clients to be attached to the server, and ``DataSet_name`` is the name of the test data.

It will read the configuration file ``appfl/config/config.yaml`` to identify its service URI.
For example, the following specifies that the URI is ``moonshot.cels.anl.gov:50051``.

.. code-block:: yaml

    server:
        id: 1
        host: moonshot.cels.anl.gov
        port: 50051

Once it is launched, it will listen to the port to serve requests from clients.
The best way to learn how to launch a server is to check out the example ``examples/grpc_mnist.py``.

For a client to connect to the server, you need to run the following function defined in ``run_grpc_client.py``:

.. code-block:: python

    run_client(cfg, comm_rank, model, train_dataset)

Its argument is defined in a similar way to that of a server, except that ``train_dataset`` represents the training data set in this case.

It will try to connect to the server specified in the ``config.yaml`` file.
We note that the server will not start training until the specified number of clients connect to the server.
We refer to ``examples/grpc_mnist.py`` for an example of running a client.
