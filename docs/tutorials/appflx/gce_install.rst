Endpoint Installation
=====================

This page describes how to install a `Globus Compute <https://globus-compute.readthedocs.io/en/stable/>`_ compute endpoint for federated learning (FL) clients on their local computing machines.

.. note::

	The Globus Compute endpoint is only supported on Linux machines. Please check this :doc:`documentation <gce_install_win>` for installing the endpoint on a Windows machine using WSL.

.. note::

	Globus Compute uses port 443 for communication, so you need to make sure your firewall allows outbound traffic on port 443.

Conda environment
-----------------

We highly recommend creating a new conda virtual environment in which to install all the packages required for setting up a Globus Compute endpoint.

.. code-block:: bash

	conda create -n appfl python=3.10 -y
	conda activate appfl

Package Installation
--------------------

You just need to install the ``appfl`` package in your conda environment, either from PyPI or from source.

.. code-block:: bash

	pip install "appfl[examples]"
	# or
	git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
	cd APPFL
	pip install -e ".[examples]"


Create Globus Account
---------------------

If you do not have a Globus account yet, please create one `here <https://app.globus.org/>`_.

.. note::

	If you can find your organization in Globus, it is highly recommended to use your organization account to log in to Globus as this makes it easier for your collaborators to verify your identity. Otherwise, you can register a Globus account using your commonly used email address.

Set Up a Globus Compute Endpoint
--------------------------------

Set up a Globus Compute endpoint using the following command. Please replace ``<ENDPOINT_NAME>`` with a name of your own, such as ``anl-aws-gpu-01``.

.. note::

	You might be required to log in with Globus when configuring the endpoint. Please follow the prompts to finish the authentication steps.

.. code-block:: bash

	globus-compute-endpoint configure <ENDPOINT_NAME>

Configure the Endpoint
----------------------

The command above creates a configuration file ``$HOME/.globus_compute/<ENDPOINT_NAME>/config.yaml``. Before starting the endpoint, update this file with the appropriate configurations for the computing resource you are using. Here are a few suggestions for setting up this configuration file:

	- If you are using your own Linux machine or a virtual machine from a cloud service provider, such as AWS EC2 or Azure, you probably do not need to change most of the ``config.yaml`` file. You just need to specify the number of blocks you want to allocate to the endpoint.

	- If you are using a supercomputer that allocates resources with a scheduler such as Slurm, you can find example configurations for various supercomputers `here <https://globus-compute.readthedocs.io/en/stable/endpoints/endpoint_examples.html#>`_.

.. note::

	- If you have further questions about setting up Globus Compute endpoints, please join the `Globus Compute Slack <https://join.slack.com/t/funcx/shared_invite/zt-gfeclqkz-RuKjkZkvj1t~eWvlnZV0KA>`_ for help.

Start the Endpoint
------------------

Now you can start the endpoint using the following command, replacing ``<ENDPOINT_NAME>`` with your endpoint name. Note that the working directory of the endpoint is ``$HOME/.globus_compute/<ENDPOINT_NAME>/tasks_working_dir``.

.. code-block:: bash

	$ globus-compute-endpoint start <ENDPOINT_NAME>

.. note::

	Whenever you start your endpoint, you should start it in the **created conda environment**.


Get Your Endpoint ID
--------------------

The following command prints the ID of your endpoint.

.. code-block:: bash

	$ globus-compute-endpoint list


Run A Simple Test
-----------------

To test whether you have successfully set up a Globus Compute endpoint, create a Python script (e.g., ``test.py``) by copying the following code. Put your own endpoint ID into the script, and you should see the result computed by your endpoint printed.

.. literalinclude:: ./endpoint.py
    :language: python
    :caption: Test script for testing Globus Compute endpoint setup.

Provide Your Endpoint to APPFLx Developer
-----------------------------------------

Dear APPFLx collaborators: currently, we have to ask the Globus Compute developers to manually add your endpoint ID to a specific Globus group before it can be used in experiments, and this process cannot be automated yet for security reasons. Please provide your endpoint ID to us via Slack, and we will register the endpoint for you.
