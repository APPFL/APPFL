Installation
============

This page describes how to install APPFL. The instructions apply regardless of your operating system.

Conda environment
-----------------

We highly recommend creating a new conda environment in which to install APPFL and its dependencies.

.. code-block:: bash

    conda create -n APPFL python=3.10
    conda activate APPFL

User installation
-----------------

For most users, including data scientists, this simple installation is sufficient for running the package.

.. code-block:: bash

    pip install pip --upgrade
    pip install "appfl[mpi,examples]"

.. note::

    If you do not need to use MPI for simulations, then you can install the package without the ``mpi`` option: ``pip install "appfl[examples]"``.

To install only the minimal set of dependencies, use:

.. code-block:: bash

    pip install appfl

.. note::

    ``torch`` may need to be updated manually to support CUDA. Please check GPU support in `PyTorch <https://pytorch.org>`_.

Developer installation
----------------------

Code developers and contributors may want to work on the local repository.
To set up the development environment, run:

.. code-block:: bash

    git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[dev,mpi,examples]"

.. note::

    If you do not need to use MPI for simulations, then you can install the package without the ``mpi`` option: ``pip install -e ".[dev,examples]"``.


On Ubuntu, if the installation fails (typically while building ``mpi4py``), install the OpenMPI system packages first and try again:

.. code-block:: bash

    sudo apt install libopenmpi-dev openmpi-bin openmpi-doc
