Join the APPFL-Hosted International Federation
==============================================

This tutorial demonstrates how to join the international federation hosted by the APPFL team.

.. note::
    This tutorial requires the user to have a valid `Globus <https://www.globus.org/>`_ account. If you do not have an account, please go to `Globus <https://www.globus.org/>`_ and click the "LOG IN" button in the top right corner. Then, look up your organization to see if it is already registered. If not, you can create a Globus ID or sign in with GitHub, Google, or ORCID.

Register with APPFL
-------------------

Once you have a Globus account, please join `APPFL's Discord <https://discord.com/invite/bBW56EYGUS>`_ and provide your Globus ID and email in the #international-federation channel. The APPFL team will then add you to the APPFL international federation and let you know when you are added.

.. note::
    To find your Globus ID, log in to Globus and click "SETTINGS" in the left menu bar. Your ID is shown in the Account tab.

Generate Your Credentials
-------------------------

Once the APPFL team notifies you that you have been added to the APPFL international federation, go to the `Globus Developer Console <https://app.globus.org/settings/developers>`_ and click the "APPFL International" project on the right. Then, click "APPFL International Federation" and you will see a console like the following. Please first note down the "Client UUID" shown in the console.

.. raw:: html

    <div style="display: flex; justify-content: center; width: 100%; margin: auto;">
        <div style="display: inline-block; ;">
            <img src="../_static/globus-appfl-international-console.png" alt="appfl_international_console">
        </div>
    </div>

Then, click "Add Client Secret", and a pop-up window will ask you to enter a name for the credential. You may enter any name you like. After you click "Generate Secret", you will see the generated secret, which is shown only once. Please copy the secret and save it in a safe place. You will need this secret to join the APPFL international federation.

.. warning::
    The secret is shown only once. If you lose it, you will need to delete the credential and generate a new one. Also, please do not share your secret with anyone else. The APPFL team will never ask you for your secret.

APPFL Installation
------------------

After obtaining the client UUID and secret, install the APPFL package on the machine you want to use for the APPFL international federation by following these steps:

1. Create a new conda environment (optional, but recommended):

   .. code-block:: bash

      conda create -n appfl python=3.10
      conda activate appfl

2. Install the APPFL package, either directly from PyPI or from source.

**Install Directly from PyPI**

.. code-block:: bash

   pip install "appfl[examples]"

**Install from Source**


.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[examples]"



Create a Globus Compute Endpoint
--------------------------------

Once you have installed the APPFL package, create a Globus Compute endpoint on your machine for the APPFL international federation by running the following commands with the client UUID and secret you obtained earlier.

.. code-block:: bash

    export GLOBUS_COMPUTE_CLIENT_ID=<Client UUID>
    export GLOBUS_COMPUTE_CLIENT_SECRET=<Client Secret>
    globus-compute-endpoint configure appfl-endpoint

.. note::

    You can replace "appfl-endpoint" with any name you like.

Configure the Globus Compute Endpoint
-------------------------------------

After creating the endpoint, you will be asked to configure it. If you are using a local computer or a cloud compute instance, you can usually use the default configuration. However, if you are using an HPC cluster or a Kubernetes cluster, you may need to modify the configuration file at ``~/.globus_compute/<your_endpoint_name>/config.yaml``. You can find `example configuration files <https://globus-compute.readthedocs.io/en/latest/endpoints/endpoint_examples.html>`_ for different types of clusters and the `configuration reference <https://globus-compute.readthedocs.io/en/latest/endpoints/config_reference.html>`_ in the Globus Compute documentation.

After configuring the endpoint, start it by running the following command:

.. code-block:: bash

    globus-compute-endpoint start appfl-endpoint

Then, please share the endpoint ID with the APPFL team in the #international-federation channel on Discord.
