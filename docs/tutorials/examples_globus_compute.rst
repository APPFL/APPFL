Example: Run FL Experiment using Globus Compute
===============================================

.. raw:: html

    <div style="display: flex; justify-content: center; width: 80%; margin: auto;">
        <div style="display: inline-block; ;">
            <img src="../_static/appfl-globus.png" alt="globus">
        </div>
    </div>

This tutorial describes how to run federated learning experiments with APPFL using Globus Compute as the communication backend. All the code needed for this tutorial is available in the `examples <https://github.com/APPFL/APPFL/tree/main/examples>`_ directory of the APPFL repository.

.. note::

    For more detailed information about Globus Compute, please refer to the `Globus Compute documentation <https://globus-compute.readthedocs.io/en/stable/index.html>`_.

Installation
------------

First, both the client and the server should install the APPFL package on their local machines. The following shows how to install the APPFL package from source. For more information, please refer to the `APPFL documentation <https://appfl.ai/en/latest/install/index.html>`_.

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    conda create -n appfl python=3.10 -y
    conda activate appfl
    pip install -e ".[examples]"

Creating Globus Compute Endpoint on Client Machines
---------------------------------------------------

Users can create a Globus Compute endpoint on their local machine with the following command. Note, however, that an endpoint created in this manner is only accessible to the user who created it, and is thus only suitable for testing purposes. To create a shared endpoint for a real federated learning experiment, please follow the `guidance below <#creating-shared-globus-compute-endpoint-on-client-machines>`_.

.. code-block:: bash

    globus-compute-endpoint configure appfl-endpoint


.. note::

    You can replace ``appfl-endpoint`` with any endpoint name you like.

Then you will be asked to configure the endpoint. If you are using your local computer, you can use the default configuration. If you are using an HPC or cloud machine, you need to modify the configuration file at ``~/.globus_compute/appfl-endpoint/config.yaml``. Below, we show a sample configuration file for `Polaris <https://www.alcf.anl.gov/polaris>`_:

.. code-block:: yaml

  engine:
    address:
      ifname: bond0
      type: address_by_interface
    max_workers_per_node: 1
    provider:
      account: <your_polaris_account> # Replace with your account
      cpus_per_node: 32
      init_blocks: 0
      max_blocks: 1
      min_blocks: 0
      nodes_per_block: 1
      queue: debug # Replace with other queue if needed
      scheduler_options: '#PBS -l filesystems=home:eagle:grand'
      select_options: ngpus=4
      type: PBSProProvider
      walltime: 00:30:00
      worker_init: module use /soft/modulefiles; module load conda; conda activate <your_conda_env>;
    strategy:
      max_idletime: 3600
      type: SimpleStrategy
    type: HighThroughputEngine

.. note::

    It is recommended to set ``max_idletime`` (in seconds) to a large value to prevent the Globus Compute service from shutting down the endpoint when no tasks are running.

After the configuration, you can start the endpoint with the following command:

.. code-block:: bash

    globus-compute-endpoint start appfl-endpoint

Creating Shared Globus Compute Endpoint on Client Machines
----------------------------------------------------------

To create shared Globus Compute endpoints for a real federated learning experiment among a group of trusted clients on their machines, the group of trusted users needs a "leading server" to create a shared Globus service account and share it with the other users, who then generate the credentials needed to start their endpoints. The following steps show how to generate such credentials:

1. The leading server needs to go to the `Globus Developer Console <https://app.globus.org/settings/developers/>`_, and click *Register a service account or application credential for automation*. The leader can then either register the application under an existing project or create a new project (it is highly recommended to create a new project for each federation). In the *App Registration* form that pops up, the leader just needs to give the app an arbitrary name and click *Register App*.

2. Click the created application to go to the application details page. You should see something like the image below. First, the leading server needs to note down the *Client UUID*. Then, click *Add Client Secret* to generate a client secret, and note down the *Client Secret*.

.. raw:: html

    <div style="display: flex; justify-content: center; width: 100%; margin: auto;">
        <div style="display: inline-block; ;">
            <img src="../_static/globus-registration.png" alt="globus-registration">
        </div>
    </div>

3. The leading server then goes back to the corresponding project page and clicks *Roles*, which shows information like the image below. Click *Assign New Role*. Then select *User* and enter the Globus ID/email of the trusted client users to assign them the Administrator role.

.. raw:: html

    <div style="display: flex; justify-content: center; width: 100%; margin: auto;">
        <div style="display: inline-block; ;">
            <img src="../_static/globus-registration-2.png" alt="globus-registration-2">
        </div>
    </div>

4. Once the trusted client users are added to the project, each of them also needs to go to the `Globus Developer Console <https://app.globus.org/settings/developers/>`_ and find the registered app under the corresponding project, note down the *Client UUID*, click *Add Client Secret* to generate a client secret for themselves, and note down the *Client Secret*.

5. After obtaining the *Client UUID* and *Client Secret*, each client needs to run the following commands before creating the shared endpoint:

.. code-block:: bash

    export GLOBUS_COMPUTE_CLIENT_ID=<Client UUID>
    export GLOBUS_COMPUTE_CLIENT_SECRET=<Client Secret>

.. note::

    The ``GLOBUS_COMPUTE_CLIENT_ID`` is the same for all the clients and the server, while the ``GLOBUS_COMPUTE_CLIENT_SECRET`` is unique to each client, who must generate it on the Globus Developer Console. Anyone with a valid Globus Compute client secret can use the other endpoints associated with the same Globus Compute client ID.

.. warning::

    The clients should NEVER share the *Client Secret* with any other clients.

6. Then, the clients can create the endpoints by following the same steps as `above <#creating-globus-compute-endpoint-on-client-machines>`_.

7. Note that the server machine should also set its corresponding *Client UUID* and *Client Secret* as environment variables before running the federated learning experiment.

.. note::

    A sample experiment log using four shared endpoints on ALCF's Polaris, Sophia, Aurora, and NCSA's Delta supercomputers is available `here <https://github.com/APPFL/APPFL/blob/main/docs/_static/sample_log.txt>`_.

Client Configurations
---------------------

The server needs to collect certain information from the client to run the federated learning experiment. Below is an example of a client configuration file. It is available at ``examples/resources/config_gc/mnist/clients.yaml`` in the APPFL repository (`link <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/mnist/clients.yaml>`_).

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

Note that the client configuration file actually resides on the server machine, and its contents are provided by the clients. Specifically, the client configuration file has three main parts:

- ``endpoint_id``: The Globus Compute endpoint ID of the client machine.
- ``train_configs``: The training configurations for the client, including the device for training, logging configurations, etc.
- ``data_configs``: Information about a dataloader Python file written by the client and shared with the server (located at ``dataset_path`` on the server machine). The dataloader file should contain a function (specified by ``dataset_name``) that loads the client's local private dataset when executed on the client's machine.

.. note::

    When the data loader function is executed on the client's machine, its default working directory is ``~/.globus_compute/appfl-endpoint/tasks_working_dir``.

Server Configurations
---------------------

We provide several sample server configuration files in ``examples/resources/config_gc/mnist`` of the APPFL repository (`link <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/>`_). A detailed description of the server configuration file can be found in the `APPFL documentation <https://appfl.ai/en/latest/users/server_agent.html#configurations>`_.

Note that ``client_configs.comm_configs.s3_configs`` is optional and should be set only if you want to use AWS S3 for data transmission (Globus Compute limits the data transmission size to 10 MB, so models larger than 10 MB should be transmitted using AWS S3). Specifically, set ``enable_s3`` to use AWS S3 for model transfer, set ``s3_bucket`` to the name of the S3 bucket you want to use, and set ``s3_creds_file`` to a CSV file containing the AWS credentials in the following format. Alternatively, the server can set this information via the ``aws configure`` command before running the experiment.

.. code-block:: csv

    <region>,<access_key_id>,<secret_access_key>


Running the Experiment
----------------------

We provide a sample experiment launch script at ``examples/globus_compute/run.py``, and you can run the experiment with the following command.

.. code-block:: bash

    python globus_compute/run.py

You can use this script as a reference and starting point for running your own federated learning experiments using Globus Compute as the communication backend.

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

As Globus Compute limits the size of function inputs and outputs to several megabytes, it is not suitable for transmitting large models. To address this issue, users can integrate Globus Compute with `ProxyStore <https://docs.proxystore.dev/latest/>`_, which facilitates efficient data flow in distributed computing applications.

By default, a `ProxyStore endpoint <https://docs.proxystore.dev/latest/guides/endpoints/>`_ connects to ProxyStore's cloud-hosted relay server, which uses Globus Auth for identity and access management. To use the provided relay server, users need to authenticate once per system using the following command:

.. code-block:: bash

    proxystore-globus-auth login

You can then create an endpoint using the following command:

.. code-block:: bash

    $ proxystore-endpoint configure my-endpoint # you can replace my-endpoint with any name you like
    INFO: Configured endpoint: my-endpoint <a6c7f036-3e29-4a7a-bf90-5a5f21056e39>
    INFO: Config and log file directory: ~/.local/share/proxystore/my-endpoint
    INFO: Start the endpoint with:
    INFO:   $ proxystore-endpoint start my-endpoint

.. note::

    You can change the endpoint configuration at ``~/.local/share/proxystore/my-endpoint/config.toml`` to change the maximum object size or to use your own relay server.

After creating the endpoint and finishing the configuration (if needed), start the endpoint with the following command:

.. code-block:: bash

    proxystore-endpoint start my-endpoint

.. note::

  For debugging the endpoint, refer to the official `ProxyStore documentation <https://docs.proxystore.dev/latest/guides/endpoints-debugging/>`_.

Configure for Federated Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ProxyStore endpoints are installed on the clients/server that want to use ProxyStore to transfer model parameters, collect all the endpoint IDs and put them in both the server and client configuration files under ``comm_configs.proxystore_configs``. Note that you only need to specify this configuration for the sites that use ProxyStore to transfer model parameters, although most of the time you will want to use it for all sites.

The following shows how to configure the server configuration file. A full sample configuration file is available at ``examples/resources/config_gc/mnist/server_fedavg_proxystore.yaml`` in the APPFL repository (`link <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/mnist/server_fedavg_proxystore.yaml>`_).

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

The following shows how to configure the client configuration file. A full sample configuration file is available at ``examples/resources/config_gc/mnist/clients_proxystore.yaml`` in the APPFL repository (`link <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_gc/mnist/clients_proxystore.yaml>`_).

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

After configuring the server and client configuration files, you can run the federated learning experiment using the same script as before, providing the new paths to the configuration files.

.. code-block:: bash

    python globus_compute/run.py \
      --server_config ./resources/config_gc/mnist/server_fedavg_proxystore.yaml \
      --client_config ./resources/config_gc/mnist/clients_proxystore.yaml

Extra: Integration with ProxyStore on Polaris
---------------------------------------------

.. raw:: html

    <div style="display: flex; justify-content: center; width: 80%; margin: auto; margin-top: 30px; margin-bottom: 30px;">
        <div style="display: inline-block; ;">
            <img src="../_static/appfl-proxystore-polaris.png" alt="polaris">
        </div>
    </div>

In this section, we show how to launch a Globus Compute endpoint on ALCF's Polaris supercomputer and use a ProxyStore endpoint to transfer model parameters between the server and clients.

Prepare the ProxyStore Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the trickiest parts of Polaris is that its compute nodes do not have internet access, except for HTTP, HTTPS, and FTP through a proxy server. Therefore, **users have to start their ProxyStore endpoint on a login node** with internet access. The started endpoint acts as a proxy for data transmission traffic between the compute nodes and the ProxyStore relay server, which listens on ``http://<login_node_id>:<port>``. When you start the endpoint with the command ``proxystore-endpoint start <endpoint_name>``, the endpoint log at ``~/.local/share/proxystore/<endpoint_name>/log.txt`` should look like the following:

.. code-block:: bash

  [2025-01-30 23:43:08.113] INFO  (proxystore.endpoint.serve) :: Installing uvloop as default event loop
  [2025-01-30 23:43:08.125] WARNING (proxystore.endpoint.serve) :: Database path not provided. Data will not be persisted
  [2025-01-30 23:43:08.125] INFO  (proxystore.endpoint.serve) :: Using native app Globus Auth client
  [2025-01-30 23:43:08.126] INFO  (globus_sdk.client) :: Creating client of type <class 'globus_sdk.services.auth.client.native_client.NativeAppAuthClient'> for service "auth"
  [2025-01-30 23:43:08.127] INFO  (globus_sdk.services.auth.client.base_login_client) :: Finished initializing AuthLoginClient. client_id='a3379dba-a492-459a-a8df-5e7676a0472f', type(authorizer)=<class 'globus_sdk.authorizers.base.NullAuthorizer'>
  [2025-01-30 23:43:08.188] INFO  (globus_sdk.authorizers.refresh_token) :: Setting up RefreshTokenAuthorizer with auth_client=[instance:139892558440592]
  [2025-01-30 23:43:08.188] INFO  (globus_sdk.authorizers.renewing) :: Setting up a RenewingAuthorizer. It will use an auth type of Bearer and can handle 401s.
  [2025-01-30 23:43:08.188] INFO  (globus_sdk.authorizers.renewing) :: RenewingAuthorizer will start by using access_token with hash "f41c966eeea9ab06d4c69aa4e0219efebe70e2f3e85fd41005ee4e954ec877fd"
  [2025-01-30 23:43:08.223] INFO  (proxystore.p2p.nat) :: Checking NAT type. This may take a moment...
  [2025-01-30 23:43:08.249] INFO  (proxystore.p2p.nat) :: NAT Type:       Full-cone NAT
  [2025-01-30 23:43:08.249] INFO  (proxystore.p2p.nat) :: External IP:    140.221.112.14
  [2025-01-30 23:43:08.249] INFO  (proxystore.p2p.nat) :: External Port:  54320
  [2025-01-30 23:43:08.250] INFO  (proxystore.p2p.nat) :: NAT traversal for peer-to-peer methods (e.g., hole-punching) is likely to work. (NAT traversal does not work reliably across symmetric NATs or poorly behaved legacy NATs.)
  [2025-01-30 23:43:08.540] INFO  (proxystore.p2p.relay.client) :: Established client connection to relay server at wss://relay.proxystore.dev with client uuid=b6cfb02b-323f-4eac-8c42-20102bb0bd26 and name=my-endpoint
  [2025-01-30 23:43:08.541] INFO  (proxystore.endpoint.endpoint) :: Endpoint[my-endpoint(b6cfb02b)]: initialized endpoint operating in PEERING mode
  [2025-01-30 23:43:08.545] INFO  (proxystore.endpoint.serve) :: Serving endpoint b6cfb02b-323f-4eac-8c42-20102bb0bd26 (my-endpoint) on 10.201.0.56:8767
  [2025-01-30 23:43:08.545] INFO  (proxystore.endpoint.serve) :: Config: name='my-endpoint' uuid='b6cfb02b-323f-4eac-8c42-20102bb0bd26' port=8767 host='10.201.0.56' relay=EndpointRelayConfig(address='wss://relay.proxystore.dev', auth=EndpointRelayAuthConfig(method='globus', kwargs={}), peer_channels=1, verify_certificate=True) storage=EndpointStorageConfig(database_path=None, max_object_size=100000000)
  [2025-01-30 23:43:08.909] INFO  (uvicorn.error) :: Started server process [909609]
  [2025-01-30 23:43:08.909] INFO  (uvicorn.error) :: Waiting for application startup.
  [2025-01-30 23:43:08.909] INFO  (proxystore.p2p.manager) :: PeerManager[my-endpoint(b6cfb02b)]: listening for messages from relay server
  [2025-01-30 23:43:08.909] INFO  (proxystore.endpoint.endpoint) :: Endpoint[my-endpoint(b6cfb02b)]: listening for peer requests
  [2025-01-30 23:43:08.910] INFO  (uvicorn.error) :: Application startup complete.
  [2025-01-30 23:43:08.910] INFO  (uvicorn.error) :: Uvicorn running on http://10.201.0.56:8767 (Press CTRL+C to quit)


.. note::

  It is important to check the log to make sure that the endpoint has started. For example, the port your endpoint is listening on might already be in use, causing an error like: ``[Errno 98] error while attempting to bind on address ('10.201.0.56', 8765): address already in use``.

Prepare the Globus Compute Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After starting the ProxyStore endpoint on a Polaris login node, you can create a Globus Compute endpoint with the following configuration. **Note that, compared with the configuration above, we specifically unset the** ``http_proxy/HTTP_PROXY`` **environment variables so that the compute node can access the ProxyStore endpoint on the login node.**

.. code-block:: yaml

  engine:
    address:
      ifname: bond0
      type: address_by_interface
    max_workers_per_node: 1
    provider:
      account: <your_polaris_account> # Replace with your account
      cpus_per_node: 32
      init_blocks: 0
      max_blocks: 1
      min_blocks: 0
      nodes_per_block: 1
      queue: debug # Replace with other queue if needed
      scheduler_options: '#PBS -l filesystems=home:eagle:grand'
      select_options: ngpus=4
      type: PBSProProvider
      walltime: 00:30:00
      worker_init: module use /soft/modulefiles; module load conda; conda activate <your_conda_env>; export HTTP_PROXY=""; export http_proxy="";
    strategy:
      max_idletime: 3600
      type: SimpleStrategy
    type: HighThroughputEngine

After the configuration, you can start the Globus Compute endpoint and configure the FL experiments as described in the previous sections.

Additional Debugging Tips
~~~~~~~~~~~~~~~~~~~~~~~~~

**Test Local ProxyStore Endpoint**:

To test if your local ProxyStore endpoint (e.g., ``my-endpoint``) is working, you can use the following command to check whether a random object exists in the endpoint store, which is expected to return ``False``.

.. code-block:: bash

  $ proxystore-endpoint test my-endpoint exists abcdef
  # Expected output
  INFO: Object exists: False

**Test Remote ProxyStore Endpoint**:

Suppose you have an endpoint running on system A with UUID ``aaaa0259-5a8c-454b-b17d-61f010d874d4`` and name ``endpoint-a``, and another on system B with UUID ``bbbbab4d-c73a-44ee-a316-58ec8857e83a`` and name ``endpoint-b``. To test the peer connection between the two endpoints from system A, you can request the endpoint on system A to invoke an ``exists`` operation on the endpoint on system B via the following command:

.. code-block:: bash

  $ proxystore-endpoint test --remote bbbbab4d-c73a-44ee-a316-58ec8857e83a endpoint-a exists abcdef
  # Expected output
  INFO: Object exists: False

.. note::

  For more detailed endpoint debugging tips, refer to the official `ProxyStore documentation <https://docs.proxystore.dev/latest/guides/endpoints-debugging/>`_.
