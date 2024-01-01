<p align="center">
  <a href="http://appfl.rtfd.io"><img src="https://github.com/APPFL/APPFL/blob/main/docs/_static/logo/logo_small.png?raw=true" alt="APPFL logo" style="width: 40%; height: auto;"></a>
</p>

<p align="center" style="font-size: 18px;">
    <b>APPFL - Advanced Privacy-Preserving Federated Learning Framework</b>.
</p>

<p align="center">
  <a href="https://zenodo.org/badge/latestdoi/414722606" target="_blank">
      <img src="https://zenodo.org/badge/414722606.svg" alt="DOI">
  </a> 
  <a href="https://appfl.readthedocs.io/en/latest/?badge=latest" target="_blank">
      <img src="https://readthedocs.org/projects/appfl/badge/?version=latest" alt="Doc">
  </a>
  <a href="https://github.com/APPFL/APPFL/actions/workflows/build.yml" target="_blank">
      <img src="https://github.com/APPFL/APPFL/actions/workflows/build.yml/badge.svg?branch=main&event=push" alt="Build">
  </a>
</p>


Advanced Privacy-Preserving Federated Learning (APPFL) is an open-source software framework that allows research communities to implement, test, and validate various ideas for privacy-preserving federated learning (PPFL).
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
On Ubuntu:
If the install process failed, you can try:
```shell
sudo apt install libopenmpi-dev,libopenmpi-bin,libopenmpi-doc
```

## APPFL Framework Design
<p align="center">
  <img src='docs/_static/design.jpg' style="width: 50%; height: auto;"/>
</p>

In the design of APPFL framework, we decompose an execution of federated learning experiment into three main components, *APPFL Server*, *APPFL Communicator*, and *APPFL Client*. The details and sub-components of these three are detailed as follows:

- APPFL Server: APPFL server orchestrates the whole FL experiment run by providing the model architecture, loss and metric, and configurations used in the training, and aggregating the client trained models synchronously or asynchronously using certain federated learning algorithms.
    
    - Model Zoo [[examples/models]](examples/models/) - This folder contains model architectures used in the given examples, and users can define their own arch for use.
    - Loss [[examples/losses]](examples/losses/) - This folder contains commonly used loss function in ML, and users can define their own loss by inheritting `nn.Module`.
    - Metric [[examples/metric]](examples/metric/) - This folder contains commonly used evaluation metric for checking the performance of the model.
    - Configuration - As shown in all the example scripts in the `examples` directory, users can setup the configurations and hyperparameters by passing arguments to the FL experiment runs.
    - FL-Alg Zoo [[src/appfl/algorithm]](src/appfl/algorithm/) - This folder contains serveral popular FL aggregation algorithms. 
- APPFL Communicator [[src/appfl/comm]](src/appfl/comm/): The communicator is used for exchanging metadata as well as the model weights between the server and clients. We currently support the following three communication protocols for different use cases.
    
    - MPI - Used for simulating FL experiment on one machine or HPC. MPI communicator now also supports model compression for efficient communication.
    - gRPC - Used for both simulating FL experiments on one machine or HPC, and running FL experiments on distributed machines.
    - [Globus Compute](https://funcx.readthedocs.io/en/latest/index.html) - Used for **easily** running FL experiments on distributed and heterogeneous machines.
- APPFL Client: APPFL clients have local dataset on disk and and use dataloader to load them for the local trainer to train local models.
    - Private Data - Each client should have on-premise private local datasets on their computing machine. For example scripts in the `examples` directory, we use `examples/datasets/RawData` to store the private datasets.
    - Dataloader [[examples/dataloader]](examples/dataloader/) - Dataloader is used to load the datasets from the storage for the trainer to use. The dataloaders in `examples/dataloader` provide a test dataset and several training datasets (one for each client) for the federated learning experiments in the MPI setting.
    - Trainer Zoo [[src/appfl/algorithm]](src/appfl/algorithm/) - This folder contains serveral commonly used local training algorithms, such as algorithm for training a model for a certain number of epochs or batches and algorithm for training a personalized FL model.

## Citation
If you find APPFL useful for your research or development, please consider citing the following papers:
```
@inproceedings{ryu2022appfl,
  title={APPFL: open-source software framework for privacy-preserving federated learning},
  author={Ryu, Minseok and Kim, Youngdae and Kim, Kibaek and Madduri, Ravi K},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
  pages={1074--1083},
  year={2022},
  organization={IEEE}
}

@inproceedings{li2023appflx,
  title={APPFLx: Providing Privacy-Preserving Cross-Silo Federated Learning as a Service},
  author={Li, Zilinghan and He, Shilan and Chaturvedi, Pranshu and Hoang, Trung-Hieu and Ryu, Minseok and Huerta, EA and Kindratenko, Volodymyr and Fuhrman, Jordan and Giger, Maryellen and Chard, Ryan and others},
  booktitle={2023 IEEE 19th International Conference on e-Science (e-Science)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
