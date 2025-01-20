Example: Run FL Experiment using Globus Compute
===============================================

.. raw:: html

    <div style="display: flex; justify-content: center; width: 80%; margin: auto;">
        <div style="display: inline-block; ;">
            <img src="../_static/appfl-globus.png" alt="globus">
        </div>
    </div>

This tutorial describes how to run federated learning experiments on APPFL using Globus Compute as the communication backend. All the code snippets needed for this tutorial is available at the ``examples`` directory of the APPFL repository at `here <https://github.com/APPFL/APPFL/tree/main/examples>`_.

.. note::

    For more detailed information about Globus Compute, please refer to the `Globus Compute documentation <https://globus-compute.readthedocs.io/en/stable/index.html>`_.

Installation
------------

First, both the client and the server should install the APPFL package on their local machines. Below shows how to install the APPFL package from its source code. For more information, please refer to the `APPFL documentation <https://appfl.ai/en/latest/install/index.html>`_.

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    conda create -n appfl python=3.10 --y
    conda activate appfl
    conda install mpi4py --y
    pip install -e ".[examples]"

Creating Globus Compute Endpoint on Client Machines
---------------------------------------------------

User can create a Globus Compute endpoint on their local machine by the following command:

.. code-block:: bash

    globus-compute-endpoint configure appfl-endpoint


.. note::

    You can replace ``appfl-endpoint`` any endpoint name you like.

Then you will be asked to configure the endpoint. If you are using your local computer, you can use the default configuration. If you are using an HPC or cloud machine, you need to modify the configuration file at ``~/.globus-compute/appfl-endpoint/config.yaml``. Below, we show a sample configuration file for `Polaris <https://www.alcf.anl.gov/polaris>`_:

.. code-block:: yaml

    display_name: appfl-endpoint

    engine:
        type: GlobusComputeEngine
        max_workers_per_node: 1

    address:
        type: address_by_interface
        ifname: bond0

    provider:
        type: PBSProProvider

        account: ParaLLMs
        queue: debug
        cpus_per_node: 32
        select_options: ngpus=4

        scheduler_options: '#PBS -l filesystems=home:eagle:grand'

        worker_init: module use /soft/modulefiles ; module load conda; conda activate appfl;

        walltime: 00:30:00
        nodes_per_block: 1
        init_blocks: 0
        max_blocks: 2
        min_blocks: 0

After the configuration, you can start the endpoint by the following command:

.. code-block:: bash

    globus-compute-endpoint start appfl-endpoint

Client Configurations
---------------------

The server needs to collect certain information from the client to run the federated learning experiment. Below is an example of a client configuration file. It is available at ``examples/resources/configs_gc/clients.yaml`` at the APPFL repository at `here <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/mnist/clients.yaml>`_.

.. code-block:: yaml

  clients:
    - endpoint_id: "ed4a1881-120e-4f67-88d7-876cd280feef"
      client_id: "Client1"
      train_configs:
        # Device [Optional]: default is "cpu"
        device: "cpu"
        # Logging and outputs [Optional]
        logging_output_dirname: "./output"
        logging_output_filename: "result"

      # Local dataset
      data_configs:
        dataset_path: "./resources/dataset/mnist_dataset.py"
        dataset_name: "get_mnist"
        dataset_kwargs:
          num_clients: 2
          client_id: 0
          partition_strategy: "class_noniid"
          visualization: False

    - endpoint_id: "762629a0-f3b3-44b5-9acf-2f9b0ab9310f"
      client_id: "Client2"
      train_configs:
        # Device [Optional]: default is "cpu"
        device: "cpu"
        # Logging and outputs [Optional]
        logging_output_dirname: "./output"
        logging_output_filename: "result"

      # Local dataset
      data_configs:
        dataset_path: "./resources/dataset/mnist_dataset.py"
        dataset_name: "get_mnist"
        dataset_kwargs:
          num_clients: 2
          client_id: 1
          partition_strategy: "class_noniid"
          visualization: False

It should be noted that the client configuration file actually resides on the server machine, and the contents of the file are shared by the clients. Specifically, there are three main parts in the client configuration file:

- ``endpoint_id``: It is the Globus Compute Endpoint ID of the client machine.
- ``train_configs``: It contains the training configurations for the client, including the device to run the training, logging configurations, etc.
- ``data_configs``: It contains the information of a dataloader python file defined and shared by the clients to the server (located at ``dataset_path`` on the server machine). The dataloader file should contain a function (specified by ``dataset_name``) which can load the client's local private dataset when it is executing on the client's machine.

.. note::

    When the data loader function is executed on the client's machine, it's default working directory is ``~/.globus-compute/appfl-endpoint/tasks_working_dir``.

Server Configurations
---------------------

We have provide three sample server configuration files available at ``examples/resources/config_gc`` at the APPFL repository at `here <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/>`_. The detailed description of the server configuration file can be found in the `APPFL documentation <https://appfl.ai/en/latest/users/server_agent.html#configurations>`_.

It should be noted that ``client_configs.comm_configs.globus_compute_configs`` is optional and should be set only if the user wants to use AWS S3 for data transmission (Globus Compute limits data transmission size to 10 MB, so models larger than 10 MB should be transmitted using AWS S3). Specifically, the ``s3_bucket`` field should be set to the name of the S3 bucket that the user wants to use, and ``s3_creds_file`` is a CSV file containing the AWS credentials. The CSV file should have the following format.

.. code-block:: csv

    <region>,<access_key_id>,<secret_access_key>

.. note::

    The server can also set these information before running the experiment via the ``aws configure`` command.

Running the Experiment
----------------------

We provide a sample experiment launching script at ``examples/globus_compute/run.py``, and user can run the experiment by the following command.

.. code-block:: bash

    python globus_compute/run.py

User can take this script as a reference and starting point to run their own federated learning experiments using Globus Compute as the communication backend.

Extra: Integration with ProxyStore
----------------------------------

.. raw:: html

    <div style="display: flex; justify-content: center; width: 80%; margin: auto; margin-top: 30px; margin-bottom: 30px;">
        <div style="display: inline-block; ;">
            <img src="../_static/appfl-proxystore.png" alt="proxystore">
        </div>
    </div>

Prepare the ProxyStore Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As Globus Compute limits the data transmission size for the function inputs and outputs to several Megabytes, it is not suitable for transmitting large models. To address this issue, users can integrate Globus Compute with `ProxyStore <https://docs.proxystore.dev/latest/>`_, which facilitates efficient data flow in distributed computing applications.

By default, a ProxyStore endpoint connects to ProxyStore's cloud-hosted relay server, which uses Globus Auth for identity and access management. To use the provided relay server, users need to do a one-time-per-system authentication using the following command:

.. code-block:: bash

    proxystore-globus-auth login

User can then create an endpoint using the following command:

.. code-block:: bash

    $ proxystore-endpoint configure my-endpoint # you can replace my-endpoint with any name you like
    INFO: Configured endpoint: my-endpoint <a6c7f036-3e29-4a7a-bf90-5a5f21056e39>
    INFO: Config and log file directory: ~/.local/share/proxystore/my-endpoint
    INFO: Start the endpoint with:
    INFO:   $ proxystore-endpoint start my-endpoint

.. note::

    User can change endpoint configuration at ``~/.local/share/proxystore/my-endpoint/config.toml`` to  change maximum object size or use their own relay server.

After creating the endpoint and finishing the configuration (if needed), user can start the endpoint by the following command:

.. code-block:: bash

    proxystore-endpoint start my-endpoint

Configure for Federated Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ProxyStore endpoints installed on the client/server which would like to use ProxyStore to transfer model parameters, user needs to collect all endpoints ids and put them in the both the server and client configuration files as ``comm_configs.proxystore_configs``. It should be noted that you only need to specify such configuration for site that you would like to use ProxyStore to transfer model parameters, although you would like to use it for all sites most of the time.

Below shows how to configure the server configuration file. A full sample configuration file is available at ``examples/resources/configs_gc/server_fedavg_proxystore.yaml`` in the APPFL repository at `here <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/mnist/server_fedavg_proxystore.yaml>`_.

.. code-block:: yaml

    client_configs:
      ... # general client configurations

    server_configs:
      ...
      comm_configs:
        proxystore_configs:
          enable_proxystore: True
          connector_type: "EndpointConnector"
          connector_configs:
            endpoints: ["endpoint_id_1", "endpoint_id_2", ...] # List of all endpoint ids for server and clients

Below shows how to configure the client configuration file. A full sample configuration file is available at ``examples/resources/configs_gc/clients_proxystore.yaml`` in the APPFL repository at `here <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/mnist/clients_proxystore.yaml>`_.

.. code-block:: yaml

    clients:
      - endpoint_id: ...
        ...
        comm_configs:
          proxystore_configs:
            enable_proxystore: True
            connector_type: "EndpointConnector"
            connector_configs:
              endpoints: ["endpoint_id_1", "endpoint_id_2", ...] # List of all endpoint ids for server and clients

      - endpoint_id: ...
        ...
        comm_configs:
          proxystore_configs:
            enable_proxystore: True
            connector_type: "EndpointConnector"
            connector_configs:
              endpoints: ["endpoint_id_1", "endpoint_id_2", ...] # List of all endpoint ids for server and clients

Running the Experiment
~~~~~~~~~~~~~~~~~~~~~~~

After configuring the server and client configuration files, user can run the federated learning experiment using the same script as before by providing the new paths to the configuration files.

.. code-block:: bash

    python globus_compute/run.py \
      --server_config ./resources/config_gc/mnist/server_fedavg_proxystore.yaml \
      --client_config ./resources/config_gc/mnist/clients_proxystore.yaml
