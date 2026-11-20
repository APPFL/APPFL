# ⚙️ Dataset
This directory contains the necessary files to load different datasets for different clients in FL experiments.

You can define your own loader files for your custom dataset by following the same manner.

## Building the data once per process

`ClientAgent` executes the loader file afresh for every client it creates. Loaders that split one dataset among the clients, such as `mnist_dataset.py` and `cifar10_dataset.py`, therefore build their test set and their partition with `appfl.misc.data.cache_per_process`: the first client created in a process builds them, and later clients in that process reuse them instead of partitioning the whole dataset again. You can do the same in your own loader, as long as the data depends only on the loader's arguments. A loader that draws unseeded random numbers while building the data, like the random crops and flips applied during the partition in `cifar10_dimat.py`, cannot cache that part.

## Download instructions for datasets

For some of the datasets used in the `examples` folder, such as `MNIST` and `CIFAR10`, they are directly available from `torchvision` and will be automatically downloaded when you first use them. However, for other datasets, there are not available through the `torchvision` package, so you need to download them manually by following the instructions below.

### Coronahack

- First, create a subdirectory named `RawData/Coronahack` under this `examples/dataset` directory.
- Download the dataset from [kaggle](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset) and store it (i.e., the `archive` directory) in the `examples/dataset/RawData/Coronahack/archive` directory.
- Preprocess the dataset by splitting it into different client chunks and resizing the image by using the preprocess scripts provided in `examples/dataset/PreprocessedData`. You can go to the directory and run the following command, where you can specify the number of client splits and the pixel size of the resized images.

    ```bash
    python Coronahack_Preprocess.py --num_clients 4 --num_pixel 32
    ```

### FEMNIST

- Make a subdirectory named `RawData/FEMNIST` under this `examples/dataset` directory.
- Clone the LEAF repository.
    ```bash
    git clone https://github.com/TalwalkarLab/leaf.git
    ```
- Go to `leaf/data/femnist` directory and run the following command, which downloads a small-sized dataset.
    ```bash
    ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
    ```
- In the newly generated directory `leaf/data/femnist/data`, copy the two directories `train` and `test` and paste them in `examples/datasets/RawData/FEMNIST`.


## CelebA
- Make a subdirectory named `RawData/CELEBA` under this `examples/dataset` directory.
- Download and preprocess the dataset according to the LEAF instructions: https://github.com/TalwalkarLab/leaf/tree/master/data/celeba
- Copy the directories `train` , `test` and `raw` folders and paste them under `examples/datasets/RawData/CELEBA`.

### FLamby

- [FLamby](https://github.com/owkin/FLamby) is a benchmark for cross-silo federated learning with seven naturally distributed real-world datasets. To use FLamby, please following the installation instructions from the [official Github](https://github.com/owkin/FLamby?tab=readme-ov-file#installation) of FLamby (Please use the installation options via pip to install `flamby` in the same conda env as `appfl`). For example, you can install the Fed-TCGA dataset by running the following commands:

    ```bash
    git clone https://github.com/owkin/FLamby.git
    cd FLamby
    pip install -e ".[tcga]"
    ```
