# APPFL

APPFL is a privacy-preserving federated learning framework that provides an infrastructure to implement algorithmic components for federated learning.

## How to run

APPFL can run in either serial or parallel with MPI on CPU/GPU architectures.
The configuration of APPFL can be found in `appfl/config` directory, where you can configure algorithm, model, privacy, compute architecture, etc.

Example of serial run to train MINIST can be done by

```
python appfl/run.py num_clients=3 num_epochs=10 device=cpu
```

where arguments `num_clients`, `num_epochs`, and `device` are optional to change their default values.

Its parallel run can be done by

```
mpiexec -np 3 --mca opal_cuda_support 1 python appfl/run.py
```

where `--mca opal_cuda_support 1` is optional to run CUDA-aware MPI.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
