Our first run: MNIST
====================

We should be ready to run the first example for APPFL. 
We first make sure that the dependencies are install and change the directory to `examples` directory.

.. code-block:: console

    $ pip install "appfl[examples]"
    $ cd examples

The first example can be simply run by this:

.. code-block:: console

    $ python mnist.py

If you want to run it in parallel using MPI (assuming that all clients are trained in the same cluster), we can run the same example as

.. code-block:: console

    $ mpiexec -np 2 python mnist.py


Our first run is training MNIST in a federated learning setting with the default configuration setting given in ``src/appfl/config/config.yaml``.
We learn more about :ref:`How to set configuration`.
