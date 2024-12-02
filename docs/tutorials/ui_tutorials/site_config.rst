Site Configuration
==================

When a user receives and accepts the group invitation to join a federation, the user becomes a site, namely, a client in federated learning. This page describes how the client user can register their computing resources and the loader of local private data to the federation via the web application.

1. Log in to the `web application <https://appflx.link>`_ by following the instructions.

2. You will be directed to a dashboard page after signing in. The dashboard lists your **Federations** and your **Sites**. Specifically, federation refers to the FL group that you created, namely, you are the group leader who can start FL experiments and access the experiment results. Site refers to the FL group of which you are a member. The federation leader is also a site of his own federation by default.

3. Click **Configure** button next to the site for which you want to register your computing resources and dataloader.

4. If you have already installed a funcX endpoint on your computing resource, just enter the endpoint ID to **Endpoint ID**. If you have not installed a funcX endpoint, either follow the instruction in the site configuration page or `here <https://ppflaas.readthedocs.io/en/latest/installation/index.html>`_. 

5. For **Git Repository Path**, please the enter the **absolute** path of the directory where you clone the APPFLx client package, such as `/home/username/appflx`. If you forget this path, you can obtain it by running the following commands on your computing device.

.. code-block:: console

    $ conda activate funcx(YOUR_ENV_NAME)
    $ pip list | grep appfl

6. For **Dataloader**, you need to provide a python script which loads your local private data by returning a PyTorch dataset (``torch.utils.data.Dataset``) containing the samples and labels for your local data. This dataloader will be called in the root directory of your funcX endpoint, so you may need to specify the relative path from the funcX root directory to your data to load the data.

7. For the dataloader file, you need to provide a ``.py`` script which contains a function defined in the following way. We provide a test `dataloader <https://github.com/APPFL/APPFLx-doc/blob/main/tutorials/mnist_dataloader.py>`_ for testing purposes, which loads part of the MNIST dataset for each client. New users are recommended to upload this test dataloader for test first.

.. literalinclude:: /tutorials/dataloader.py
    :language: python
    :caption: Template for local dataloader: tutorials/dataloader.py


.. note::

	Though you upload a dataloader for your private and sensitive local data, it is only called on your own computing resource for local training and no training data will leave your own computing resources. 

7. When you have your dataloader file ready, you can either upload it from your local computer by clicking **Upload from Computer** or upload it from Github by clicking **Upload from Github**. When you choose to upload from Github, a modal will pop up, first click **Authorize with Github** to link your Github account, then you can choose or search for the repository, select the branch and file to upload.

8. For **Device Type**, select **cpu** if your computing device does not have GPU or you don't want to use GPU in trianing, otherwise, select **cuda:0** to enable GPU usage in training.

9. Click **Save** to save the configuration for your computing resources and local private data. 


