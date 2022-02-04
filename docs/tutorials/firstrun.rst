Our first run: MNIST
====================

We present how to run an example script for APPFL. 
We first make sure that the dependencies are installed and change the directory to `examples` directory.

.. code-block:: console

    $ git clone https://github.com/APPFL/APPFL.git
    $ cd APPFL
    $ pip install -e ".[examples]"
    $ cd examples

The first example can be simply run by this:

.. code-block:: console

    $ python mnist.py

If we want to run it in parallel using MPI (assuming that all clients are trained in the same cluster), we can run the same example as

.. code-block:: console

    $ mpiexec -np 5 python mnist.py

We can also simulate PPFL with gRPC.

.. code-block:: console

    $ mpiexec -np 5 python grpc_mnist.py

.. note::

    ``mpiexec`` may need to specify additional argument to use CUDA: ``--mca opal_cuda_support 1``

Our first run is training MNIST in a federated learning setting with the default configuration.
Learn more about :ref:`How to set configuration`.
