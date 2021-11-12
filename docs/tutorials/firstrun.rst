Your first run
==============

You should be ready to run the first example for APPFL. The first example can be simply done by this:

.. code-block:: console

    $ python appfl/run

You can run it in parallel using MPI (assuming that all clients are trained in the same cluster):

.. code-block:: console

    $ mpiexec -np 2 python appfl/run


What did you just run?
----------------------

The APPFL run is defined in and reads the following configuration file:

.. literalinclude:: /../appfl/config/config.yaml
    :language: YAML
    :caption: Configuration file: appfl/config/config.yaml

Please find details about the syntax for the configuration file in `Hydra <https://hydra.cc>`_.
