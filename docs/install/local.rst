Local
=====

We recommend to create a Conda virtual environment and install the required packages for APPFL.

.. code-block:: console

    $ conda create -n APPFL python=3.8
    $ conda activate APPFL

Then, install `PyTorch <https://pytorch.org>`_. 

.. note::
    
    The best way is to use the installation command from the PyTorch website, depending on your architecture.

We also need to install the following python packages:

.. code-block:: console

    $ pip install hydra-core --upgrade
    $ pip install mpi4py --upgrade
    $ pip install matplotlib --upgrade
    $ pip install tensorboard --upgrade

Optionally, you can install Jupyter Notebook.

.. code-block:: console

    $ pip install notebook

