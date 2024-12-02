Check Experiment Results
========================

This page describes how the federation leader can check the results of federated learning experiments conducted under certain federation.

.. note::
	
	Currently, only the group leader of a certain federation can check the experiment results.

1. Log in to the `web application <https://appflx.link>`_ by following the instructions. 

2. You will be directed to a dashboard page after signing in. The dashboard lists your **Federations** and your **Sites**. Specifically, federation refers to the FL group that you created, namely, you are the group leader who can start FL experiments and access the experiment results. Site refers to the FL group of which you are a member. The federation leader is also a site of his own federation by default.

3. Click the name of the federation for which you want to check the experiment results, which will lead you to the **Federation Information** page. 

4. **Endpoint Information** at the top of the Federation Information page shows the information of the clients and the status of their computing resources (funcX endpoints).

5. **Experiment Information** lists the information of all experiments conducted under this federation. For a certain experiment, click the **Config** icon to see the configurations for that experiment, click the **Log** icon to see the training log of the orchestration server, click **Report** icon to see the generated report for the experiment, and click the **Tensorboard** icon to see the tensorboard log of the experiment.

6. You can select multiple experiments by clicking the check boxes to the left of the experiment names, and you can click the **Compare** button to compare the results of multiple experiments.

.. note::
	
	Currently, the **delete** button is not available for use yet.