Example: Run FL Experiment using Ray
====================================

This tutorial describes how to run federated learning experiments with APPFL using Ray on cloud platforms such as AWS, GCP, and Azure. All the code needed for this tutorial is available in the `examples <https://github.com/APPFL/APPFL/tree/main/examples>`_ directory of the APPFL repository.

.. note::

    For more detailed information about Ray, please refer to the `Ray documentation <https://docs.ray.io/en/latest/index.html>`_.

Installation
------------

First, install the APPFL package on your local machine. The following shows how to install the APPFL package from source. For more information, please refer to the `APPFL installation guide <https://appfl.ai/en/latest/install/index.html>`_.

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    conda create -n appfl python=3.10 -y
    conda activate appfl
    pip install -e ".[examples]"

Client Configurations
---------------------

The server needs to collect certain information from the client to run the federated learning experiment. Below is an example of a client configuration file. It is available at ``examples/resources/config_ray/mnist/clients.yaml`` in the APPFL repository (`link <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_ray/mnist/clients.yaml>`_).

.. code-block:: yaml

    clients:
      - client_id: "Client1"
        train_configs:
          # Device [Optional]: default is "cpu"
          device: "cpu"
          # Logging [Optional]
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
            visualization: True

      - client_id: "Client2"
        train_configs:
          # Device [Optional]: default is "cpu"
          device: "cpu"
          # Logging [Optional]
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

- ``client_id``: The unique ID defined for the client machine.
- ``train_configs``: The training configurations for the client, including the device for training, logging configurations, etc.
- ``data_configs``: Information about a dataloader Python file written by the client and shared with the server (located at ``dataset_path`` on the server machine). The dataloader file should contain a function (specified by ``dataset_name``) that loads the client's local private dataset when executed on the client's machine.


Server Configurations
---------------------

We provide several sample server configuration files in ``examples/resources/config_ray/mnist`` of the APPFL repository (`link <https://github.com/APPFL/APPFL/blob/main/examples/resources/config_ray/>`_). A detailed description of the server configuration file can be found in the `APPFL documentation <https://appfl.ai/en/latest/users/server_agent.html#configurations>`_.

Note that ``client_configs.comm_configs.ray_configs`` is optional and only needs to be set if you want to assign client tasks to random worker nodes instead of particular AWS instances, by setting ``assign_random`` to ``True`` (``False`` by default). In that case, you need to configure ``ray_cluster_config.yaml`` accordingly as well.

To use AWS S3 for model parameter transmission, add a configuration under ``comm_configs`` as ``s3_configs``. Set ``enable_s3`` to ``True``, and specify the ``s3_bucket`` field with the name of the S3 bucket that you want to use. Additionally, set ``s3_creds_file`` to the path of a CSV file containing AWS credentials.

.. code-block:: csv

    <region>,<access_key_id>,<secret_access_key>

.. note::

    The server can also set this information via the ``aws configure`` command before running the experiment.

Ray Cluster Configurations
--------------------------

Below is the cluster configuration file for running the experiment on AWS.

.. code-block:: yaml

    # A unique identifier for the head node and workers of this cluster.
    cluster_name: appfl-ray

    # Cloud-provider specific configuration.
    provider:
        type: aws
        region: us-east-1
        cache_stopped_nodes: False # if set False terminates the instance when ray down is executed, True: instance stopped not terminated
        security_group:
            GroupName: ray_client_security_group
            IpPermissions:
                - FromPort: 8265
                  ToPort: 8265
                  IpProtocol: TCP
                  IpRanges:
                      # Allow traffic only from your local IP address.
                      - CidrIp: 0.0.0.0/0

    # The maximum number of worker nodes to launch in addition to the head node.
    max_workers: 2

    available_node_types:
        ray.head.default:
            resources: { }
            # Provider-specific config for this node type, e.g., instance type.
            # By default Ray auto-configures unspecified fields such as SubnetId and KeyName.
            # For more documentation on available fields, see
            # http://boto3.readthedocs.io/en/latest/reference/services/ec2.html#EC2.ServiceResource.create_instances
            node_config:
                InstanceType: t3.medium
                ImageId: 'ami-0dd6adfad4ad37eec' # Deep Learning Base Neuron AMI (Ubuntu 20.04) 20240216
        ray.worker.worker_1:
            # The minimum number of worker nodes of this type to launch.
            # This number should be >= 0. For FL experiment 1 is sufficient.
            min_workers: 1
            # The maximum number of worker nodes of this type to launch.
            # This parameter takes precedence over min_workers. For FL experiment 1 is sufficient.
            max_workers: 1
            # Set this to {${client_id} : 1}, client_id from examples/resources/config_ray/mnist/clients.yaml config file
            # Set it to empty if client task can be assigned randomly to any worker node
            resources: {Client1: 1}
            node_config:
                InstanceType: t3.medium
                ImageId: 'ami-0dd6adfad4ad37eec' # Deep Learning Base Neuron AMI (Ubuntu 20.04) 20240216
                InstanceMarketOptions:
                    MarketType: spot  # Configure worker nodes to use Spot Instances
                    SpotOptions:
                        MaxPrice: '0.05'
        ray.worker.worker_2:
            min_workers: 1
            max_workers: 1
            resources: {Client2: 1}
            node_config:
                InstanceType: t3.medium
                ImageId: 'ami-0dd6adfad4ad37eec' # Deep Learning Base Neuron AMI (Ubuntu 20.04) 20240216
                InstanceMarketOptions:
                    MarketType: spot  # Configure worker nodes to use Spot Instances
                    SpotOptions:
                        MaxPrice: '0.05'

    file_mounts: {
        "/home/ubuntu/APPFL": "../../../APPFL",
        "/home/ubuntu/resources": "../resources",
        "/home/ubuntu/run.py": "run.py"
    }

    setup_commands:
        ["conda config --remove channels intel",
         "conda create -n APPFL python=3.10 -y ",
         'conda activate APPFL && pip install ray["default"] && pip install confluent-kafka --prefer-binary && cd APPFL && pip install -e ".[examples]"',
         "(stat $HOME/anaconda3/envs/APPFL/ &> /dev/null && echo 'export PATH=\"$HOME/anaconda3/envs/APPFL/bin:$PATH\"' >> ~/.bashrc) || true"]

You can set the desired AWS region under ``provider.region``.

All EC2 instance-related configurations for the head node or worker nodes go in ``node_config``, which includes ``InstanceType``, ``ImageId`` (AMI image ID), spot vs. on-demand, etc. For more documentation on the available fields, see `here <http://boto3.readthedocs.io/en/latest/reference/services/ec2.html#EC2.ServiceResource.create_instances>`_.

For descriptions of the other fields, see the inline comments in ``examples/ray/ray_cluster_config.yaml``, or check out the Ray documentation `here <https://docs.ray.io/en/latest/cluster/vms/getting-started.html#launch-a-cluster-on-a-cloud-provider>`_.


Running Experiment
------------------

Environment setup
~~~~~~~~~~~~~~~~~

Configure AWS credentials for an IAM user with the ``AmazonEC2FullAccess`` and ``AmazonEC2RoleforSSM`` policies.

Cluster Creation
~~~~~~~~~~~~~~~~

Go to the Ray example directory:

.. code-block:: bash

    cd examples/ray/

Run the following command, which brings up the whole cluster described in ``examples/ray/ray_cluster_config.yaml``.

.. code-block:: bash

    ray up ray_cluster_config.yaml

.. note::

    To reduce the cluster spin-up time, create a custom AMI by running the setup commands on the image specified in ``ray_cluster_config.yaml``. After creating the custom AMI, set it as the ``ImageId`` attribute of each node in ``ray_cluster_config.yaml``.

Checking cluster status
~~~~~~~~~~~~~~~~~~~~~~~

**From Local Machine**

1. Check the cluster status by running:

.. code-block:: bash

    ray exec ray_cluster_config.yaml 'ray status'

**From Head Node**

1. Attach to the head node using:

.. code-block:: bash

    ray attach ray_cluster_config.yaml

2. After attaching to the head node, check the cluster status using:

.. code-block:: bash

    ray status

The output of ``ray status`` should look like the following:

.. code-block:: bash

    ======== Autoscaler status: 2025-02-25 20:18:02.106153 ========
    Node status
    ---------------------------------------------------------------
    Active:
     1 ray.worker.worker_2
     1 ray.head.default
     1 ray.worker.worker_1
    Pending:
     (no pending nodes)
    Recent failures:
     (no failures)

    Resources
    ---------------------------------------------------------------
    Usage:
     0.0/6.0 CPU
     0.0/1.0 Client1
     0.0/1.0 Client2
     0B/7.64GiB memory
     0B/3.16GiB object_store_memory

    Demands:
     (no resource demands)


Job Submission
~~~~~~~~~~~~~~

**From Local Machine**

1. Set up port forwarding using:

.. code-block:: bash

    ray dashboard ray_cluster_config.yaml

2. In another terminal, submit the job using:

.. code-block:: bash

    ray job submit --address http://localhost:8265  -- python APPFL/examples/ray/run.py

**From Head Node**

1. Connect to the head node:

.. code-block:: bash

    ray attach ray_cluster_config.yaml

2. Run the job using:

.. code-block:: bash

    python run.py

Stopping Cluster
~~~~~~~~~~~~~~~~
1. To stop the cluster, run:

.. code-block:: bash

    ray down ray_cluster_config.yaml
