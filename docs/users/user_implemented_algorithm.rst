Implemented FL Algorithms 
==========================

Various global model update methods @ server
-------------------------------------------
We have implemented various global model update methods taken place at the server given local model parameters received from clients, which are

- ``ServerFedAvg``            : averaging local model parameters to update global model parameters      
- ``ServerFedAvgMomentum``    : ServerFedAvg with a momentum        
- ``ServerFedAdagrad``        : use of the adaptive gradient (Adagrad) algorithm for a global update at a server
- ``ServerFedAdam``           : use of the adaptive moment estimation (Adam) algorithm for a global update 
- ``ServerFedYogi``           : use of the Yogi algorithm for a global update         

One can set which algorithm to use by setting ``servername`` in ``federated.py`` (e.g., ``cfg.fed.servername=ServerFedAvg``).
One can configure each algorithm (e.g., selecting hyperparameters) in ``federated.py``.

.. literalinclude:: /../src/appfl/config/fed/federated.py
    :language: python
    :lines: 31-41
    :caption: configuration of the global update methods

Roughly speaking, the global update is done as follow:
 
``global_model_parameter`` += (``server_learning_rate`` * ``m``) / ( sqrt(``v``) + ``server_adapt_param``)

where

``server_learning_rate``: learning rate for the global update

``server_adapt_param``: adaptivity parameter 

``m``: momentum 

- ``m`` = ``server_momentum_param_1`` * ``m`` + (1- ``server_momentum_param_1``) * ``PseudoGrad``
- ``PseudoGrad`` : pseudo gradient obtained by averaging differences of global and local model parameters

``v``: variance
  
- For ``ServerFedAdagrad``:      
    ``v`` = ``v`` + (``PseudoGrad``)^2  
- For ``ServerFedAdam``:
    ``v`` = ``server_momentum_param_2`` * ``v`` + (1- ``server_momentum_param_2``)* (``PseudoGrad``)^2
- For ``ServerFedYogi``:  
    ``v`` = ``v`` - (1 - ``server_momentum_param_2`` )* (``PseudoGrad``)^2 sign(``v`` - (``PseudoGrad``)^2)
           

See the following paper for more details on the above global update techniques:

Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S. and McMahan, H.B., 2020. Adaptive federated optimization. arXiv preprint arXiv:2003.00295

Client optimizers
------------------
Clients can choose which optimizer to use for their local training.
To this end, one can select optimizer supported by PyTorch (e.g., SGD, Adam, or LBFGS) in ``federated.py`` and its corresponding arguments.

.. literalinclude:: /../src/appfl/config/fed/federated.py
    :language: python
    :lines: 42-47
    :caption: configuration of the client optimizers

As an example of using LBFGS, one can simply add the followings in the example file (e.g., see ``mnist_no_mpi.py``):

.. code-block:: python
    
    parser.add_argument("--client_optimizer", type=str, default="LBFGS")

    if args.client_optimizer == "LBFGS":        
        cfg.fed.clientname = "ClientOptimClosure" ## LBFGS requires to reevalute functions for multiple times (have to pass in a closure)
        cfg.batch_training = False ## mini-batch training is not supported by the vanilla LBFGS 
        cfg.fed.args.optim_args.lr = 10.0  
        cfg.fed.args.optim_args.max_iter=10000 
        cfg.fed.args.optim_args.tolerance_grad=1e-10 ality (default: 1e-5).
        cfg.fed.args.optim_args.tolerance_change=1e-10 
        cfg.fed.args.optim_args.history_size=1000 
        cfg.fed.args.optim_args.line_search_fn="strong_wolfe" 