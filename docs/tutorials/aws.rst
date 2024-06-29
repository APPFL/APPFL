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

Generating SSL Certificates for Secure gRPC Connections on EC2
----------------------

1. Intall OpenSSL and Verify Installation

.. code-block:: shell

    sudo apt update
    sudo apt install openssl
    openssl version

.. note::
    
    If you find that the subsequent steps do not generate valid certificates, try changing the openssh version. The version I use for this Tutorial is OpenSSL 1.1.1w.

2. Generate a Private Key

First, a private key file (.key file) is generated for signing certificates.

.. code-block:: shell

    openssl genpkey -algorithm RSA -out server.key

3. Generate a Certificate Signing Request (CSR) [Optional]

If you intend to send a certificate signing request to a certificate authority (CA), you can generate a certificate request (CSR) file. This step is optional and can be skipped if you intend to generate a self-signed certificate.

.. code-block:: shell

    openssl req -new -key server.key -out server.csr \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=example.com"

.. note::
    
    You can change content in -subj based on your info. /C means countr; /ST means State; /L means City; /O means organization; And /CN means Common Name.

4. Generate the SSL Certificate

Finally, a self-signed certificate (.crt file) is generated using the generated private key and certificate request file (optional). In this step, we will include the Subject Alternate Name (SAN) to cover different access scenarios (public network and private network).

For Self-Signed Certificate:

.. code-block:: shell

    openssl req -x509 -days 365 -key server.key -out server.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=example.com" \
    -addext "subjectAltName = DNS:your.domain.name"

.. note::
    
    Subject Alternate Name (SAN): In the -addext parameter, use -addext "subjectAltName = DNS:your.domain.name" to add a subject alternate name. Be sure to replace your.domain.name with the domain name or host name you wish to use as the SAN. For example, the Public IPv4 DNS of your EC2.
    Certificate validity and key length: Depending on your security needs, you can adjust the validity of the certificate and the length of the generated key.

CA-Signed Certificate:
If you prefer to have your CSR signed by a CA, you would send the server.csr file to the CA and receive a signed certificate in return. The exact process depends on the CA's requirements.

5. Configure gRPC to Use SSL
Once you have the server.key and server.crt files, you can configure your gRPC server to use them for SSL encryption.
