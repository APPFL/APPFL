How to run PPFL
===============

``APPFL`` provides users with the capabilities of simulating and training PPFL on either a single machine, a cluster, or multiple heterogeneous machines.
We refer

- **simulation** as running PPFL on a single machine or a cluster without actual data decentralization
- **training** as running PPFL on multiple (heterogeneous) machines with actual decentralization of client datasets

Hence, we describe two types of PPFL run:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   user_simulation
   user_train


:ref:`Simulating PPFL` is useful for those who develop, test, and validate new models and algorithms for PPFL, whereas :ref:`Training PPFL` for those who consider actual PPFL settings in practice.

Sample template
---------------

.. note::

    Before reading this section, please check out :ref:`Tutorials` for more detailed examples in notebooks.


For either simulation or training, a skeleton of the script for running PPFL can be written as follows:

.. code-block:: python
    :linenos:

    from appfl import *
    from appfl.config import *

    def main():

        # load default configuration
        cfg: DictConfig = OmegaConf.structured(Config)

        model = ... # user-defined model
        data = ... # user-defined datasets

        # The choice of PPFL runs

    if __name__ == "__main__":
        main()


Some remarks are made as follows:

- Line 7 loads the default configuration for PPFL run. See :ref:`How to set configuration` for more details.
- User-defined model and data can be read as in lines 9 and 10; see :ref:`User-defined model` and :ref:`User-defined dataset`.
- Depending on our choice of PPFL run, we then call API functions in line 12.
