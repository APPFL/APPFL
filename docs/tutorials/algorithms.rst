Incorporating your algorithm
============================

In a federated learning setting, a server updates a global model parameter based on the local model parameters updated by multiple clients.

How to incorporate your algorithm into our framework APPFL?

1. Create classes for the global and local updates in ``appfl/algorithm``
2. Create a configuration file that specifies the algorithm in ``appfl/config/fed``
3. Global update is conducted in a ``run_server`` function in ``appfl/run.py``

Code snippet:

.. code-block:: python     

    server.update(global_state, local_states)

4. Local update is conducted in a ``run_client`` function in ``appfl/run.py``

Code snippet:

.. code-block:: python     

    client.update()   

 
**Example. Federated Averaging (FedAvg)** 

1. Create classes for the global and local updates in ``appfl/algorithm``

   - See two classes ``FedAvgServer`` and ``FedAvgClient`` in ``fedavg.py``
   - In ``FedAvgServer``, the ``update`` function conducts a global update by averaging the local model parameters sent from multiple clients
   - In ``FedAvgClient``, the ``update`` function conducts a local update and send the resulting local model parameters to the server

2. Create a configuration file that specifies the algorithm in ``appfl/config/fed``

   - The algorithm is specified in ``fedavg.yaml``

