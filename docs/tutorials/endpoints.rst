How to run with Endpoints for Compute Engine with ESP
=====================================================

This describes how to deploy a simple example `gRPC <https://grpc.io>`_ server with the Extensible Server Proxy (ESP) through a Docker container in `Google Compute Engine <https://cloud.google.com/compute>`_.

This page is based on this Google's `documentation <https://cloud.google.com/endpoints/docs/grpc/get-started-compute-engine-docker>`_ with more detailed references. In this page we describes the steps specific for the example with ``appfl``.

As prerequisites, we need create a `Compute Engine instance <https://cloud.google.com/endpoints/docs/grpc/get-started-compute-engine-docker#create_vm>`_


Configuring Endpoints
---------------------

As this service requires `.proto` file, we first clone the package repository:

.. code-block:: shell

    git clone git@github.com:APPFL/APPFL.git


The example we use in this page is located in ``APPFL/examples/gcloud``.

.. code-block:: shell

    cd APPFL/examples/gcloud


Creating a self-contained protobuf descriptor file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We create a protobuf descriptor file from ``../../src/appfl/protos/federated_learning.proto`` file.
To this end, we create the following directory

.. code-block:: shell

    mkdir generated_pb2


and create the descriptor file ``api_descriptor.pb`` by running the command below:

.. code-block:: shell

    python -m grpc_tools.protoc \
        --include_imports \
        --include_source_info \
        --proto_path=../../src/appfl/protos \
        --descriptor_set_out=api_descriptor.pb \
        --python_out=generated_pb2 \
        --grpc_python_out=generated_pb2 \
        federated_learning.proto


Creating a gRPC API configuration YAML file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need to create the configuration file ``api_config.yaml``. We provide the simple working example below:

.. code-block:: yaml

    type: google.api.Service
    config_version: 3

    name: appfl.endpoints.<YOUR_PROJECT_ID>.cloud.goog

    title: APPFL gRPC API
    apis:
    - name: FederatedLearning


The value of ``name`` becomes the API service name, and the value of ``apis.name`` should match with the service name defined in ``.proto`` file.


Deploying the Endpoints configuration
-------------------------------------

Follow the steps in `this page <https://cloud.google.com/endpoints/docs/grpc/get-started-compute-engine-docker#deploy_configuration>`_.


Deploying the API backend
-------------------------

Follow the steps to `install Docker on the VM instance <https://cloud.google.com/endpoints/docs/grpc/get-started-compute-engine-docker#install_docker_on_the_vm_instance>`_.

We provide the the Docker commands below to run the example API and ESP in a Docker container, customized from the `Google documentation <https://cloud.google.com/endpoints/docs/grpc/get-started-compute-engine-docker#running_the_sample_api_and_esp_in_a_docker_container>`_. The following commands are run on the Google Compute Engine VM instance.

1. On the VM instance, create a container network

    .. code-block:: shell

        sudo docker network create --driver bridge esp_net

2. Run the APPFL gRPC server Docker container with the name of ``appfl-test`` on the network ``esp_net``. The last line replaces the default ``CMD`` value defined in the Docker container.
   
    .. code-block:: shell

        sudo docker run \
        --detach \
        --name=appfl-test \
        --net=esp_net \
        gcr.io/$GCP_PROJECT/appfl-test:latest \
        python3 grpc_mnist_server.py --nclients=3


3. Run the ESP Docker container provided by Google. By ``--publish``, port 80 will be exposed and connected to HTTP2 port 9000.

    .. code-block:: shell

        sudo docker run \
        --detach \
        --name=esp \
        --publish=80:9000 \
        --net=esp_net \
        gcr.io/endpoints-release/endpoints-runtime:1 \
        --service=appfl.endpoints.<YOUR_PROJECT_ID>.cloud.goog \
        --rollout_strategy=managed \
        --http2_port=9000 \
        --backend=grpc://appfl-test:50051


Launching a client
------------------

Find the external IP address for the gRPC Endpoints:

.. code-block:: shell

    gcloud compute instances list


Launching a client to connect to the Endpoints requires to use `API key <https://cloud.google.com/docs/authentication/api-keys>`_. Follow the steps to create an API key to run a client.
A client can start the federated learning by running the command below with the API key ``copyandpasteyourapikeyhere``:

.. code-block:: shell

  python grpc_mnist_client.py \
  --host=<EXTERNAL_IP_ADDRESS> \
  --port=80 \
  --client_id=1 \
  --nclients=3 \
  --api_key=copyandpasteyourapikeyhere