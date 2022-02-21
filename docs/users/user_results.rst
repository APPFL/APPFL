Save FL Results and Models
==========================

APPFL store FL outputs (i.e., ``result.txt`` and ``model.pt``) in a ``examples/outputs`` directory.

- ``result.txt``: It displays how FL takes place per every communication rounds. 

.. code-block:: console

      Iter     Local[s]    Global[s]     Valid[s]      Iter[s]   Elapsed[s]  TestAvgLoss TestAccuracy     Prim_res     Dual_res    Penal_min    Penal_max 
         1        13.05         0.02         2.05        15.12        15.12     0.174392        94.32   8.9469e+00   0.0000e+00         0.00         0.00 
         2        12.63         0.02         1.92        14.57        29.69     0.089002        97.15   3.1964e+00   0.0000e+00         0.00         0.00 


The contents can be modified by revising ``print_write_result_title``, ``print_write_result_iteration``, and ``print_write_result_summary`` in ``src/appfl/misc/utils.py``

- ``model.pt``: Trained model is saved in a ``TorchScript`` format (for more details, see https://pytorch.org/tutorials/beginner/saving_loading_models.html). 


Note that the name of the directory, i.e., ``examples/outputs``, and filename, i.e., ``result.txt`` and ``model.pt``, can be modified by revising ``output_dir``, ``result_name``, and ``model_name`` in ``src/appfl/config/config.py``. 