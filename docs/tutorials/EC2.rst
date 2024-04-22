How to setup EC2 instance
=======================

This describes how to set up the gRPC server to run on `Amazon EC2 <https://aws.amazon.com/ec2/>`_, while the clients are training the model locally.


Creating EC2 instance and Configuring
------------------------------------

1. Sign in to AWS Management Console: Sign in to the AWS Management Console using your AWS account.

2. Select EC2 service: On the console home page, find and click on "EC2" or search for and select "EC2" in the "Services" menu.

3. Start the instance:
    a. In the EC2 console, click "Instances" and select "Launch Instance".  

    b. Name the instance.

    c. Select `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type` as the Amazon Machine Image (AMI); the AMI determines your instance's operating system and pre-installed software. 

    d. Select `t2.medium` as instance type. The instance type determines the CPU, memory, storage, and network capacity of the virtual server. Please choose one that is at least as good as or better than t2.medium.  

    e. Generate key pari for later logging.

    f. Configure storage. Please allocate at least 45GB of storage space for the EC2.

    g. The rest of the configuration is left as default. And click `Launch Instance` to launch.

4. Assigning an Elastic IP Address

    To ensure that the instance has a fixed public IP address, you can assign an elastic IP address:
    
    a. Go to the AWS Management Console and select the "EC2" service.
    
    b. Select "Elastic IP" in the left navigation bar.

    c. Click "Assign New Address" and follow the prompts to get an Elastic IP.

    d. After assigning an Elastic IP, select the IP and click the "Actions" button to select "Associate Address".

    e. Select the instance you want to associate with and confirm.

5. Setting Up Security Groups

    To allow external machines to connect to your EC2 instance via gRPC, you need to configure security groups to allow inbound TCP traffic on the corresponding ports:

    a. Log in to the AWS Management Console and go to the "EC2" service.

    b. In the EC2 console, find and select the security group associated with your instance.

    c. In the security group details, select the "Inbound Rules" tab and click "Edit Inbound Rule".

    d. Add a new rule, select "Custom TCP" as the protocol type, set the port range to the port used by your gRPC service (e.g. 50051), and set the source to the allowed IP range (e.g. 0.0.0.0/0 means anywhere, but should be limited as much as possible for security reasons).

    e. Click "Save Rule".


Connecting EC2 instance and Deploying
------------------------------------

There are several methods to connect EC2 instance. You can select EC2 instance and click connect to connect using EC2 instance connect provided by AWS.

If you want to connect with SSH client. Follow the steps below.

.. code-block:: shell

  cd `location of your pem`

.. code-block:: shell

  chmod 400 "your_pem_name.pem"

.. code-block:: shell

  ssh -i "your_pem_name.pem" ubuntu@Public_DNS_of_your_instance

After successfully connecting to the EC2 instance, install Conda environment first. Pls do not install the environment directly on EC2. Errors may pop out.

Follow the commands below to setup conda.

.. code-block:: shell

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

.. code-block:: shell

  chmod +x Miniconda3-latest-Linux-x86_64.sh

.. code-block:: shell

  ./Miniconda3-latest-Linux-x86_64.sh

.. code-block:: shell

  source ~/miniconda3/bin/activate

.. code-block:: shell

  conda info

Now Conda environment is set up in your EC2 instance. We can create environment and install fedcompass.

Pls follow the installation instructions in `FedCompass <https://github.com/APPFL/FedCompass>`_ to install fedcompass.


Launching a server
------------------

A server can be started with the following:

.. code-block:: shell

  python FedCompass/examples/grpc/run_server.py 


.. note::

  You may need to change config of grpc server to fit your condition. The config of grpc server is in examples/config/server_fedcompass.yaml.
