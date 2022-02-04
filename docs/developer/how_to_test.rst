How to test code
================

We use ``PyTest`` to run test scripts in ``tests``.  Any new development should be extensively tested.
The test assumes that three MPI processes are used:

.. code-block:: shell

    $ mpirun -n 3 python -m pytest --with-mpi
    