Your first run: MNIST
=====================

You should be ready to run the first example for APPFL. 
First, change the directory to `examples` directory.

.. code-block:: console

    $ cd examples

The first example can be simply done by this:

.. code-block:: console

    $ python mnist.py

If you want to run it in parallel using MPI (assuming that all clients are trained in the same cluster), we can run the same example as

.. code-block:: console

    $ mpiexec -np 2 python mnist.py


What did you just run?
----------------------

The APPFL run is defined in and reads the following configuration file:

.. literalinclude:: /../src/appfl/config/config.yaml
    :language: YAML
    :caption: Configuration file: src/appfl/config/config.yaml

Please find details about the syntax for the configuration file in `Hydra <https://hydra.cc>`_.
