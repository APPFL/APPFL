On GPU Cluster (Argonne's Swing)
================================

This describes how to set up the environment to run APPFL in GPU cluster. This tutorial is generated based on SWING GPU cluster in Argonne National Laboratory. The cluster information is avaiable at `Laboratory Computing Resource Center <https://www.lcrc.anl.gov/systems/resources/swing/>`_. In this tutorial, we use MNIST example to run APPFL in the cluster. 

Preparing Training
--------------------------------
We assume user run the MNIST example in locally machine according to `Our first run MNIST <https://github.com/APPFL/APPFL/blob/main/docs/tutorials/firstrun.rst>`_. MNIST datasets will be downloaded while running the MNIST example.

We upload the data and example code from local machine to cluster.

.. code-block:: console

	cd APPFL/examples
	ssh [your_id]@[cluster_destination] mkdir -p workspace	 
	scp -r * [your_id]@[cluster_destination]:workspace	

Please check if the workspace folder contains "datasets", "mnist.py", "models" for this tutorial.

Loading Modules
------------------------------------
This tutorial uses `modules <https://hpc-wiki.info/hpc/Modules>`_ in SWING cluster. The module configuration may vary depending on the Clusters. 

.. code-block:: console

	module load gcc/9.2.0-r4tyw54 cuda/11.4.0-gqbcqie openmpi/4.1.4-cuda-ucx anaconda3

Creating Conda Environment and Installing APPFL
-----------------------------------------------
Anaconda environment is used to control dependencies.

.. code-block:: console

	conda create -n APPFL python=3.8
	conda activate APPFL
	pip install pip --upgrade	
	pip install "appfl[dev,examples,analytics]"


Modifying Dependencies for CUDA Support
---------------------------------------------
SWING Cluster uses CUDA 11.4 version, so we need to modify torch version to adjust to the CUDA version. CUDA version may vary depending on the Clusters. A different version of CUDA may require changing the `torch <https://pytorch.org/get-started/locally/>`_ versions.

.. code-block:: console

	pip uninstall torch tourchvision	
	pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
	conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

.. Note::

	``pip install chardet`` may need to resolve the dependency issue from the torchvision package.

Creating Batch Script
---------------------------------------------
SWING Cluster uses Slurm workload manager for job management. The job management configuration may vary depending on the Clusters. 


.. code-block:: console
	:caption: test.sh

	#!/bin/bash
	#
	#SBATCH --job-name=APPFL-test
	#SBATCH --account=<your_project_name>
	#SBATCH --nodes=1
	#SBATCH --gres=gpu:2
	#SBATCH --time=00:05:00

	mpiexec -np 2 --mca opal_cuda_support 1 python ./mnist.py --num_clients=2

The script needs to be submitted to run.

.. code-block:: console

	sbatch test.sh

You may see the output.

.. code-block:: console

	Submitted batch job {job_id}

The output file is generated when the script run.

.. code-block:: console

	cat slurm-{job_number}.out
	

