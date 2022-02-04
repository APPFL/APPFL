Installation
============

This page describes how to install APPFL on a machine independent to operating systems.
Machine-specific installation steps may be added later.

Conda environment
-----------------

We recommend to create new Conda virtual environment and install the required packages for APPFL.

.. code-block:: console

    $ conda create -n APPFL python=3.8
    $ conda activate APPFL

User installation
-----------------

For most users, including data scientists, this simple installation must be sufficient for running the package.

.. code-block:: console

    $ pip install pip --upgrade
    $ pip install "appfl[analytics]"

If you want to even minimize the installation of package dependencies, you can use:

.. code-block:: console

    $ pip install appfl

.. note::

    ``torch`` may need to be updated manually to supprt CUDA. Please check GPU support in `PyTorch <pytorch.org>`_.

Developer installation
----------------------

Code developers and contributors may want to work on the local repository. 
To set up the development environment, 

.. code-block:: console

    $ git clone https://github.com/APPFL/APPFL.git
    $ cd APPFL
    $ pip install -e ".[dev,examples,analytics]"
