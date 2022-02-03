How to run PPFL
===============

``APPFL`` provides users with the capabilities of simulating and training PPFL on either a single machine, a cluster, or multiple heterogeneous machines.
We describe two types of PPFL run:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_simulation
   user_train


We create a python script by using API functions in ``appfl`` package.

.. code-block:: python
    :linenos:

    import appfl.run as ppfl
    from appfl.config import *
    from omegaconf import DictConfig

    def main(cfg: DictConfig):

        # user-defined model
        # user-defined datasets
        # The choice of PPFL runs

    if __name__ == "__main__":
        main()


Some remarks are made as follows:

- We use ``Hydra`` and ``omegaconf`` packages to read the configuration files in YAML. See :ref:`How to set configuration` for details. The main configuration file (``config.yaml``) should be located at the path given in line 5 of the example code.
- User-defined model and data can be read as in lines 7-8; see :ref:`How to define a model` and :ref:`How to define local dataset`.
- The serial simulation run will start by ``run_serial`` with the configuration, user-defined model, and user datasets, as given in line 10. The last argument gives the name of dataset.
