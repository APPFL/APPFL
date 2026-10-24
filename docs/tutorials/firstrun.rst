Quickstart
==========

On this page, we show how to run an example federated learning script on the MNIST dataset using the APPFL package.

Installation
------------

First, make sure that the APPFL package and its dependencies are installed. Then, change to the ``examples`` directory.

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[mpi,examples]"
    cd examples


.. note::

    If you do not need to use MPI for simulations, then you can install the package without the ``mpi`` option: ``pip install -e ".[examples]"``.

Serial simulation
-----------------

The first example can be run with the following command, which launches a federated learning experiment with five clients. The federated learning server and the five clients run serially on one machine for simulation.

.. code-block:: bash

    python ./serial/run_serial.py --num_clients 5 \
        --server_config ./resources/configs/mnist/server_fedavg.yaml \
        --client_config ./resources/configs/mnist/client_1.yaml

.. note::

    1. ``--server_config`` specifies the path to the configuration file for the FL server. In this example, we use the FedAvg algorithm as the server algorithm. Note that only synchronous FL algorithms can be used for serial experiments, as it does not make sense to run asynchronous algorithms serially.
    2. ``--client_config`` specifies the path to the base configuration file for all FL clients.
    3. ``--num_clients`` specifies the number of federated learning clients.

MPI simulation
--------------

To run the FL experiment in parallel using MPI, use the following command, which runs the `FedCompass <https://arxiv.org/pdf/2309.14675.pdf>`_ algorithm with five clients.

.. code-block:: bash

    mpiexec -n 6 python ./mpi/run_mpi.py --server_config ./resources/configs/mnist/server_fedcompass.yaml \
        --client_config ./resources/configs/mnist/client_1.yaml

.. note::

    1. ``-n 6`` in the above command means that we are launching 6 MPI processes. As a result, the number of clients is equal to the number of processes minus one (one MPI process is used as the FL server), which is 5 in this case.
    2. You can easily run different FL algorithms (synchronous and asynchronous) by changing the server configuration file. For example, you can run the `FedAsync <https://arxiv.org/pdf/1903.03934.pdf>`_ algorithm by changing the server configuration file to ``./resources/configs/mnist/server_fedasync.yaml``.

MPI simulation with privacy
---------------------------

As the package name suggests, in addition to general differential privacy techniques, APPFL also provides two special synchronous privacy-preserving algorithms, `IIADMM <https://arxiv.org/pdf/2202.03672.pdf>`_ and `ICEADMM <https://arxiv.org/pdf/2110.15318.pdf>`_. You can run these privacy-preserving federated learning (PPFL) algorithms with the following commands.

.. code-block:: bash

    mpiexec -n 6 python ./mpi/run_mpi_admm.py --server_config ./resources/configs/mnist/server_iiadmm.yaml
    # OR
    mpiexec -n 6 python ./mpi/run_mpi_admm.py --server_config ./resources/configs/mnist/server_iceadmm.yaml

.. note::

    Compared with ``mpi/run_mpi.py``, ``mpi/run_mpi_admm.py`` has the following additional lines, which let the client obtain its relative sample size (weight) and pass it to the client agent, as required for local training.

    .. code-block:: python

        # (Specific to ICEADMM and IIADMM) Send the sample size to the server and set the client weight
        sample_size = client_agent.get_sample_size()
        client_weight = client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size, sync=True)
        client_agent.trainer.set_weight(client_weight["client_weight"])


gRPC deployment
---------------

To show how to deploy the APPFL package on a real distributed system, we provide an example of running the federated learning experiment on the MNIST dataset using gRPC as the communication protocol.

First, run the following command to start a federated learning server using the ``FedCompass`` algorithm.

.. code-block:: bash

    python ./grpc/run_server.py --config ./resources/configs/mnist/server_fedcompass.yaml

Open a second terminal and start a client that connects to the server using the following command.

.. code-block:: bash

    python ./grpc/run_client.py --config ./resources/configs/mnist/client_1.yaml

Open a third terminal and start another client using the following command.

.. code-block:: bash

    python ./grpc/run_client.py --config ./resources/configs/mnist/client_2.yaml
