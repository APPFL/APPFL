How to simulate PPFL
====================

This package provides users with the capability of simulating PPFL on either a single machine or a cluster.

.. note::

    Running (either training or simulating) PPFL on multiple heterogeneous machines is described in :ref:`How to train PPFL`.


We describe how to simulate PPFL with a given model and dataset. 


.. code-block:: python

    import appfl.run as ppfl
    import hydra
    from omegaconf import DictConfig

    # Define model, train_data (for each client), and test_data

    def main(cfg: DictConfig):
        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        train_data, test_data = get_data(comm)
        model = get_model(comm)

        if comm_size > 1:
            if comm_rank == 0:
                ppfl.run_server(cfg, comm, model, test_data, num_clients, DataSet_name)
            else:
                ppfl.run_client(cfg, comm, model, train_data, num_clients)
        else:
            ppfl.run_serial(cfg, model, train_data, test_data, DataSet_name)

    @hydra.main(config_path="./config", config_name="config")
    def main(cfg: DictConfig):
        ppfl.run_serial(cfg, model, train_data, test_data, "example")

    if __name__ == "__main__":
        main()


Serial run
----------

.. code-block:: console

    $ python ./my_appfl.py


MPI
---

.. code-block:: console

    $ mpiexec -np 5 python ./my_appfl.py