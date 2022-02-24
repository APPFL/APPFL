# APPFL: Argonne Privacy-Preserving Federated Learning

![image](https://github.com/APPFL/APPFL/blob/main/docs/_static/logo/logo_small.png?raw=true)

[![DOI](https://zenodo.org/badge/414722606.svg)](https://zenodo.org/badge/latestdoi/414722606)
[![Documentation Status](https://readthedocs.org/projects/appfl/badge/?version=latest)](https://appfl.readthedocs.io/en/latest/?badge=latest)

APPFL is an open-source software framework that allows research communities to implement, test, and validate various ideas for privacy-preserving federated learning (PPFL).
With this framework, developers and/or users can

- train a user-defined neural network model on **decentralized data with differential privacy**,
- **simulate** various PPFL algorithms on high-performance computing (HPC) architecture with MPI,
- implement **user-defined PPFL algorithms** in a plug-and-play manner.

Such algorithmic components include federated learning (FL) algorithm, privacy technique, communication protocol, FL model to train, and data.

- [Documentation](http://appfl.rtfd.io/): please check out the documentation for tutorials, users guide, and developers guide.

## Installation

We highly recommend to create new Conda virtual environment and install the required packages for APPFL.

```shell
conda create -n APPFL python=3.8
conda activate APPFL
```

### User installation

For most users such as data scientists, this simple installation must be sufficient for running the package.

```shell
pip install pip --upgrade
pip install "appfl[analytics,examples]"
```

If we want to even minimize the installation of package dependencies, we can skip the installation of a few pacakges (e.g., `matplotlib` and `jupyter`):

```shell
pip install "appfl"
```

### Developer installation

Code developers and contributors may want to work on the local repositofy. 
To set up the development environment, 

```shell
git clone https://github.com/APPFL/APPFL.git
cd APPFL
pip install -e ".[dev,examples,analytics]"
```

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
