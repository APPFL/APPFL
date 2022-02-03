Simulating PPFL
===============

This package provides users with the capability of simulating PPFL on either a single machine or a cluster.

.. note::

    Running (either training or simulating) PPFL on multiple heterogeneous machines is described in :ref:`this section <Training PPFL>`.


We describe how to simulate PPFL with a given model and datasets. For simulation, we assume that ``test_data`` is also available to validate the training.



Serial run
----------

We create a python script by using API functions in ``appfl`` package.

.. code-block:: python
    :linenos:
    :emphasize-lines: 5, 7-8, 10

    import appfl.run as ppfl
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="./config", config_name="config")
    def main(cfg: DictConfig):
        model = ...                 # user-defined model
        train_data, test_data = ... # user-defined datasets

        ppfl.run_serial(cfg, model, train_data, test_data, "my_appfl")

    if __name__ == "__main__":
        main()


Some remarks are made as follows:

- We use ``Hydra`` and ``omegaconf`` packages to read the configuration files in YAML. See :ref:`How to set configuration` for details. The main configuration file (``config.yaml``) should be located at the path given in line 5 of the example code.
- User-defined model and data can be read as in lines 7-8; see :ref:`How to define a model` and :ref:`How to define local dataset`.
- The serial simulation run will start by ``run_serial`` with the configuration, user-defined model, and user datasets, as given in line 10. The last argument gives the name of dataset.

Finally in the shell, the serial run script can be run:

.. code-block:: console

    $ python my_appfl.py


Parallel run with MPI
---------------------

The script of parallel runs can be written with much change from the serial script above.

.. code-block:: python
    :linenos:
    :emphasize-lines: 4, 11-15

    import appfl.run as ppfl
    import hydra
    from omegaconf import DictConfig
    import mpi4py

    @hydra.main(config_path="./config", config_name="config")
    def main(cfg: DictConfig):
        model = ...                 # user-defined model
        train_data, test_data = ... # user-defined datasets

        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            ppfl.run_server(cfg, comm, model, test_data, num_clients, "my_appfl")
        else:
            ppfl.run_client(cfg, comm, model, train_data, num_clients)

    if __name__ == "__main__":
        main()


To use MPI, we import ``mpi4py`` package in line 4. In the parallel run script above, we assume that MPI process 0 runs the server, and the other processes run clients. The server and the clients begin by ``run_server`` and ``run_client``, respectively, in lines 13 and 15, where MPI communicator (i.e., ``MPI.COMM_WORLD`` in this example) is given as an argument.

The parallel run sciprt can be run as follows:

.. code-block:: console

    $ mpiexec -np 5 python ./my_appfl.py