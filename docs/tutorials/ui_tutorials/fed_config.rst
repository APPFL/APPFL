Start Experiments
=================

This page describes how the FL group (federation) leader can start an FL experiment via the web application.

1. Log in to the `web application <https://appflx.link>`_ by following the instructions. 

2. You will be directed to a dashboard page after signing in. The dashboard lists your **Federations** and your **Sites**. Specifically, federation refers to the FL group that you created, namely, you are the group leader who can start FL experiments and access the experiment results. Site refers to the FL group of which you are a member. The federation leader is also a site of his own federation by default.

3. Click **Create New Experiment** button next to the federation for which you want to start the FL experiment. This will lead you to the **Federation Configuration** page.

4. **Client Endpoints** at the top of the page shows the status of client (site) funcX endpoints. Click the status icon to see the details of the endpoint status. Only clients (sites) with active endpoints will join the FL experiment. You can contact the client via email by clicking the email icon if the client endpoint is not active.

5. For **Federation Algorithm**, we support the following federated learning algorithms. Choose one algorithm that you want to use.


- `Federated Average <https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf>`_: Communication-Efficient Learning of Deep Networks from Decentralized Data

- `Federated Average Momentum <https://arxiv.org/pdf/1909.06335.pdf>`_: Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification

- `Federated Adagrad/Adam/Yogi <https://arxiv.org/pdf/2003.00295.pdf>`_: Adaptive Federated Optimization

- `Federated Asynchronous <https://arxiv.org/pdf/1903.03934.pdf>`_: Asynchronous Federated Optimization

6. For **Experiment Name**, please provide a name of your choice for this FL experiment.

7. For **Server Training Epochs**, enter the number of global aggregations for the FL experiment.

8. For **Client Training Epochs**, enter the number of local training epochs for each client (site) before sending the local model back to the server.

9. For **Server Validation Set for Benchmarking**, select **None** if you are doing your own experiments. Select **MNIST** only if your FL group is using the provided MNIST dataloader for testing purposes, and this will enable the orchestration server to download the MNIST test dataset and perform server test.

10. **Privacy Budget**, **Clip Value** and **Clip Norm** are used for preserving privacy, enter 0 to disable this.

11. Upload the training model architecture by either selecting a **Template Model**, uploading a **Custom Model**, or choosing a custom model by **Uploading from Github**. When you choose upload from Github, a modal will pop up, first click **Authorize with Github** to link your Github account, then you can choose or search for the repository, select the branch and file to upload. If you want to use a custom model, you need to provide a ``.py`` script which contains a function called ``get_model()``. You can find a template model definition file `here <https://github.com/APPFL/APPFLx-doc/blob/main/tutorials/cnn.py>`_. Basically, you need to first define your PyTorch model by inheriting ``torch.nn.Module``, and then define the ``get_model`` function to return the defined model class.

12. For **Client Optimizer**, choose either SGD or Adam, and specify the local learning rate of each client in **Client Learning Rate**. For different client local training rounds, you can choose to decay the client learning rate by entering a value between 0 and 1 in **Client Learning Rate Decay**. 

13. For **Client Weights**, **Proportional to Sample Size** means applying different weights to different client local models during the global aggregation by calculating the weights proportional to the client sample size, and **Equal for All Clients** means applying the same weights to all client local models.

14. After carefully choosing all configurations and hyperparameters for the FL experiment, you can start the experiment by clicking **Start**. Then the web application will launch an orchestration server for you which trains a federated learning model by collaborating with all active client endpoints.