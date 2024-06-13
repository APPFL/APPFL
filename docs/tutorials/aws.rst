Launch server on AWS EC2
========================

In this page, we describe the processes to set up a gRPC server for FL on an `AWS EC2 <https://aws.amazon.com/ec2/>`_ instance, which waits for clients to connect and participate in FL experiments.

Create an EC2 instance
----------------------

1. Sign in to the `AWS Management Console <https://aws.amazon.com/>`_ using your AWS account.

2. Select EC2 service: On the console home page, search for and select "EC2" in the "Services" menu.

3. Start the instance:
   
    a. In the EC2 console, click "Instances" and select "Launch Instance".  

    b. Name the instance.

    c. Select `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type` as the Amazon Machine Image (AMI); the AMI determines your instance's operating system and pre-installed software. 

    d. Select your instance type. The instance type determines the CPU, memory, storage, and network capacity of the virtual machine. Please choose one that is at least as good as or better than `t2.medium`.  

    e. Generate key pair for connecting for your instance later.

    f. Configure storage. Please allocate at least 45GB of disk space for the EC2.

    g. The rest of the configuration can be left as default. And click `Launch Instance` to launch.

4. To ensure that the instance has a fixed public IP address, you can assign an elastic IP address:

    a. Go to the AWS Management Console and select the "EC2" service.
    
    b. Select "Elastic IP" in the left navigation bar.

    c. Click "Assign New Address" and follow the prompts to get an Elastic IP.

    d. After assigning an Elastic IP, select the IP and click the "Actions" button to select "Associate Address".

    e. Select the instance you want to associate with and confirm.

5. Set Up Security Groups

    To allow external machines to connect to your EC2 instance via gRPC, you need to configure security groups to allow inbound TCP traffic on the corresponding ports:

    a. Log in to the AWS Management Console and go to the "EC2" service.

    b. In the EC2 console, find and select the security group associated with your instance.

    c. In the security group details, select the "Inbound Rules" tab and click "Edit Inbound Rule".

    d. Add a new rule, select "Custom TCP" as the protocol type, set the port range to the port used by your gRPC service (e.g. 50051), and set the source to the allowed IP range (e.g. 0.0.0.0/0 means anywhere, but should be limited as much as possible for security reasons).

    e. Click "Save Rule".

Connect to EC2 instance and launch the server
---------------------------------------------

There are several methods to connect EC2 instance. You can select EC2 instance and click connect to connect using EC2 instance connect provided by AWS.

If you want to connect with SSH client. Follow the steps below.

.. code-block:: shell

    cd <path_to_dir_containing_pem>
    chmod 400 "your_pem_name.pem"
    ssh -i "your_pem_name.pem" ubuntu@Public_DNS_of_your_instance

.. note::
    
    In order to successfully ssh into the EC2 instance, you also need to allow the IP address of the machine you are using to connect to the EC2 instance for ssh connection. 

After successfully connecting to the EC2 instance, install conda.

.. code-block:: shell

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    source ~/miniconda3/bin/activate

Installation
------------

Install APPFL from source in a conda environment:

.. code-block:: console

    git clone https://github.com/APPFL/APPFL.git
    cd APPFL
    conda create -n appfl python=3.10 --y
    conda activate appfl
    pip install -e ".[examples, compressor]"

Launching a server
------------------

An FL server running the `FedCompass <https://arxiv.org/pdf/2309.14675.pdf>`_ algorithm can be started with the following:

.. code-block:: shell

    cd examples
    python grpc/run_server.py --config configs/mnist/server_fedcompass.yaml

.. note::

  You may need to change path of the configuration file of grpc server to select different FL algoirthms.