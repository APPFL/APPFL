Simulation on GPU cluster
=========================

This describes how to set up the environment to run APPFL using MPI in a GPU cluster for simulation, which is useful for benchmarking the performance of different FL algoirthms on various datasets. In this example, we partition the CIFAR10 in an non-independent and identically distributed (non-IID) manner into five client splits and train a Resnet-18 model using the FedAvg algorithm.

.. note::

	This tutorial is generated based on the `Delta supercomputer <https://docs.ncsa.illinois.edu/systems/delta/en/latest>`_ at the National Center for Supercomputing Applications (NCSA), which uses Slurm as it job scheduler. 

Loading Modules
---------------

Most HPC clusters use `modules <https://hpc-wiki.info/hpc/Modules>`_ to manage the environment, and the module configuration may vary depending on the clusters you use. On the Delta supercomputer, the following modules are loaded.

This tutorial uses `modules <https://hpc-wiki.info/hpc/Modules>`_ in SWING cluster. The module configuration may vary depending on the Clusters. 

.. code-block:: console

	1) gcc/11.4.0   2) openmpi/4.1.6   3) cuda/11.8.0   4) cue-login-env/1.0   5) slurm-env/0.1   6) default-s11   7) anaconda3_gpu/23.9.0

You need to run `module save` to save the current module configuration.

.. code-block:: bash

	module save

Creating Conda Environment and Installing APPFL
-----------------------------------------------
Now, we can create a conda environment and install APPFL.

.. code-block:: bash

    conda create -n appfl python=3.10
    conda activate appfl
    git clone https://github.com/APPFL/APPFL.git
    cd APPFL
    pip install -e ".[examples]"
    cd examples


Creating Batch Script
---------------------
The Delta supercomputer uses Slurm workload manager for job management. 

.. code-block:: bash
	:caption: submit.sh

	#!/bin/bash
	#SBATCH --mem=150g                              # required number of memory
	#SBATCH --nodes=1                               # number of required nodes
	#SBATCH --ntasks-per-node=6                    	# number of tasks per node [SHOULD BE EQUAL TO THE NUMBER OF CLIENTS+1]
	#SBATCH --cpus-per-task=1                       # <- match to OMP_NUM_THREADS
	#SBATCH --partition=gpuA40x4                    # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
	#SBATCH --account=<xxxx-delta-gpu>              # <- one of: replace xxxx with your project name
	#SBATCH --job-name=APPFL-test				    # job name
	#SBATCH --time=00:15:00                         # dd-hh:mm:ss for the job
	#SBATCH --gpus-per-node=1
	#SBATCH --gpu-bind=none

	source ~/.bashrc
	conda activate appfl
	cd <your_path_to_appfl>/examples
	mpiexec -np 6 python ./mpi/run_mpi.py --server_config ./configs/cifar10/server_fedcompass.yaml --client_config ./configs/cifar10/client_1.yaml

The script needs to be submitted to run.

.. code-block:: console

	sbatch test.sh

You may see the output.

.. code-block:: console

	Submitted batch job {job_id}

The output file `slurm-{job_id}.out` is generated when the script starts to run, and you can check the output in real-time by running the following command.

.. code-block:: console

	tail -f -n 10 slurm-{job_id}.out
