Client Configuration
====================

When a user receives and accepts the group invitation to join a federation, the user becomes a client in federated learning. This page describes how the client user can register their computing resources and the loader for their local private data with the federation via the web application.

1. Log in to the `web application <https://appflx.link>`_ by following the instructions.

2. You will be directed to a dashboard page after signing in. The dashboard lists your **Federations** and your **Clients**. Specifically, **Federations** are the FL groups that you created, i.e., groups in which you are the leader who can start FL experiments and access the experiment results. **Clients** are the FL groups of which you are a member. By default, the federation leader is also a client of their own federation.

3. Click the **Configure** button next to the client for which you want to register your computing resources and dataloader.

4. If you have already installed a Globus Compute endpoint on your computing resource, just enter the endpoint ID in **Endpoint ID**. If you have not installed a Globus Compute endpoint, follow the instructions on the site configuration page or :doc:`here <gce_install>`.

5. For **Dataloader**, you need to provide a Python script that loads your local private data and returns a PyTorch dataset (``torch.utils.data.Dataset``) containing the samples and labels. Whenever you load data from your local file system, please use absolute paths.

6. The dataloader file must be a ``.py`` script containing a function defined as described above. We provide an example for loading the MNIST dataset.

.. literalinclude:: ./mnist_dataset.py
    :language: python
    :caption: Example for local dataloader.

.. note::

	Although you upload a dataloader for your private and sensitive local data, it is only called on your own computing resource for local training, and no training data ever leaves your computing resources.

7. When your dataloader file is ready, you can either upload it from your local computer by clicking **Upload from Computer** or upload it from GitHub by clicking **Upload from Github**. If you choose to upload from GitHub, a modal will pop up: first click **Authorize with Github** to link your GitHub account, then choose or search for the repository, and select the branch and file to upload.

8. For **Device Type**, select **cpu** if your computing device does not have a GPU or you do not want to use a GPU for training; otherwise, select **cuda** to enable GPU training.

9. Click **Save** to save the configuration for your computing resources and local private data.
