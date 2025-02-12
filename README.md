<p align="center">
  <a href="http://appfl.rtfd.io"><img src="https://github.com/APPFL/APPFL/blob/main/docs/_static/logo/logo_small.png?raw=true" alt="APPFL logo" style="width: 40%; height: auto;"></a>
</p>

<p align="center" style="font-size: 18px;">
    <b>APPFL - Advanced Privacy-Preserving Federated Learning Framework</b>.
</p>

<p align="center">
  <a href="https://discord.com/invite/bBW56EYGUS">
      <img src="https://dcbadge.vercel.app/api/server/bBW56EYGUS?theme=default-inverted&style=flat" alt="discord">
  </a>
  <a href="https://zenodo.org/badge/latestdoi/414722606" target="_blank">
      <img src="https://zenodo.org/badge/414722606.svg" alt="DOI">
  </a>
  <a href="https://appfl.readthedocs.io/en/latest/?badge=latest" target="_blank">
      <img src="https://readthedocs.org/projects/appfl/badge/?version=latest" alt="Doc">
  </a>
  <a href="https://github.com/APPFL/APPFL/actions/workflows/build.yml" target="_blank">
      <img src="https://github.com/APPFL/APPFL/actions/workflows/build.yml/badge.svg?branch=main&event=push" alt="Build">
  </a>
  <a href="https://results.pre-commit.ci/latest/github/APPFL/APPFL/main">
      <img src="https://results.pre-commit.ci/badge/github/APPFL/APPFL/main.svg" alt="pre-commit">
  </a>
  <a href="https://arxiv.org/abs/2202.03672">
      <img src="https://img.shields.io/badge/arXiv-2202.03672-B31B1B.svg" alt="APPFL">
  </a>
  <a href="https://arxiv.org/abs/2409.11585">
      <img src="https://img.shields.io/badge/arXiv-2409.11585-B31B1B.svg" alt="APPFL-Advance">
  </a>
</p>

APPFL, Advanced Privacy-Preserving Federated Learning, is an open-source and highly extensible software framework that allows research communities to implement, test, and validate various ideas related to privacy-preserving federated learning (FL), and deploy real FL experiments easily and safely among distributed clients to train more robust ML models.With this framework, developers and users can easily

* Train any user-defined machine learning model on decentralized data with optional differential privacy and client authentication.
* Simulate various synchronous and asynchronous PPFL algorithms on high-performance computing (HPC) architecture with MPI.
* Implement customizations in a plug-and-play manner for all aspects of FL, including aggregation algorithms, server scheduling strategies, and client local trainers.

[Documentation](http://appfl.rtfd.io/): please check out our documentation for tutorials, users guide, and developers guide.

## Table of Contents

* [Installation](#hammer_and_wrench-installation)
* [Technical Components](#bricks-technical-components)
* [Framework Overview](#bulb-framework-overview)
* [Citation](#page_facing_up-citation)
* [Acknowledgements](#trophy-acknowledgements)

## :hammer_and_wrench: Installation

We highly recommend creating a new Conda virtual environment and install the required packages for APPFL.

```bash
conda create -n appfl python=3.8
conda activate appfl
```

### User installation

For most users such as data scientists, this simple installation must be sufficient for running the package.

```bash
pip install pip --upgrade
pip install "appfl[examples,mpi]"
```

💡 Note: If you do not need to use MPI for simulations, then you can install the package without the ``mpi`` option: ``pip install "appfl[examples]"``.

If we want to even minimize the installation of package dependencies, we can skip the installation of a few packages (e.g., `matplotlib` and `jupyter`):

```bash
pip install "appfl"
```

### Developer installation

Code developers and contributors may want to work on the local repositofy.
To set up the development environment,

```bash
git clone --single-branch --branch main https://github.com/APPFL/APPFL.git
cd APPFL
pip install -e ".[mpi,dev,examples]"
```

💡 Note: If you do not need to use MPI for simulations, then you can install the package without the ``mpi`` option: ``pip install -e ".[dev,examples]"``.

On Ubuntu:
If the install process failed, you can try:
```bash
sudo apt install libopenmpi-dev,libopenmpi-bin,libopenmpi-doc
```

## :bricks: Technical Components

APPFL is primarily composed of the following six technical components

* Aggregator: APPFL supports several popular algorithms to aggregate one or several client local models.
* Scheduler: APPFL supports several synchronous and asynchronous scheduling algorithms at the server-side to deal with different arrival times of client local models.
* Trianer: APPFL supports several client local trainers for various training tasks.
* Privacy: APPFL supports several global/local differential privacy schemes.
* Communicator: APPFL supports MPI for single-machine/cluster simulation, and gRPC and Globus Compute with authenticator for secure distributed training.
* Compressor: APPFL supports several lossy compressors for model parameters, including [SZ2](https://github.com/szcompressor/SZ), [SZ3](https://github.com/szcompressor/SZ3), [ZFP](https://pypi.org/project/zfpy/), and [SZx](https://github.com/szcompressor/SZx).

## :bulb: Framework Overview
<p align="center">
  <img src='https://github.com/APPFL/APPFL/blob/main/docs/_static/design-logic-new.jpg?raw=true' style="width: 85%; height: auto;"/>
</p>

In the design of the APPFL framework, we essentially create the server agent and client agent, using the six technical components above as building blocks, to act on behalf of the FL server and clients to conduct FL experiments. For more details, please refer to our [documentation](http://appfl.rtfd.io/).

## :page_facing_up: Citation
If you find APPFL useful for your research or development, please consider citing the following papers:
```
@article{li2024advances,
  title={Advances in APPFL: A Comprehensive and Extensible Federated Learning Framework},
  author={Li, Zilinghan and He, Shilan and Yang, Ze and Ryu, Minseok and Kim, Kibaek and Madduri, Ravi},
  journal={arXiv preprint arXiv:2409.11585},
  year={2024}
}

@inproceedings{ryu2022appfl,
  title={APPFL: open-source software framework for privacy-preserving federated learning},
  author={Ryu, Minseok and Kim, Youngdae and Kim, Kibaek and Madduri, Ravi K},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
  pages={1074--1083},
  year={2022},
  organization={IEEE}
}
```

## :trophy: Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
