# ⬇️ Instructions for Downloading Datasets
The users can store their raw datasets in this directory for further loading, and this file contains the instructions on how to download certain datasets used in the examples.

## Example 1. Coronahack

- Make a subdirectory named ``Coronahack`` under this directory.
- Download the dataset from https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset 
- Store the dataset (i.e., `archive` directory) in the ``examples/datasets/RawData/Coronahack/archive`` 
- Preprocess the dataset by splitting it into different client chunks and resizing the image. The preprocess script is provided in `examples/datasets/PreprocessedData`, and you can go to that directory and run the following command, where you can specify the number of client splits and the pixel size of the resized images.
    ```
    python Coronahack_Preprocess.py --num_clients 4 --num_pixel 32
    ```

## Example 2. FEMNIST

- Make a subdirectory named ``FEMNIST`` under this directory.
- Clone the LEAF repository.
    ```
    git clone https://github.com/TalwalkarLab/leaf.git
    ```
- Go to ``leaf/data/femnist`` directory run the following command, which downloads a small-sized dataset.
    ```
    ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
    ```
- In the newly generated directory ``leaf/data/femnist/data``, copy the two directories ``train`` and ``test`` and paste them in ``examples/datasets/RawData/FEMNIST``.
 

## Example 3. CelebA
- Make a subdirectory named ``CELEBA`` under this directory.
- Download and preprocess the dataset according to the LEAF instructions: https://github.com/TalwalkarLab/leaf/tree/master/data/celeba
- Copy the directories ``train`` , ``test`` and ``raw`` folders and paste them under ``examples/datasets/RawData/CELEBA``