Check Experiment Results
========================

This page describes how the federation leader can check the results of federated learning experiments conducted within a federation.

.. note::

	Currently, only the group leader of a federation can check its experiment results.

1. Log in to the `web application <https://appflx.link>`_ by following the instructions.

2. You will be directed to a dashboard page after signing in. The dashboard lists your **Federations** and your **Clients**. Specifically, **Federations** are the FL groups that you created, i.e., groups in which you are the leader who can start FL experiments and access the experiment results. **Clients** are the FL groups of which you are a member. By default, the federation leader is also a client of their own federation.

3. Click the name of the federation for which you want to check the experiment results, which will lead you to the **Group Information** page.

4. **Clients Information** at the top of the page shows the information of the clients and the status of their computing resources (Globus Compute endpoints).

5. **Experiment Information** lists all experiments conducted within this federation. For each experiment, click the **Config** icon to see the configurations for that experiment, click the **Log** icon to see the training log of the orchestration server, and click the **Report** icon to see the generated report for the experiment.

6. You can delete experiment data by selecting the check boxes to the left of the experiment names and clicking the **Delete** button.
