# Raw Data
The users can store their raw datasets in this directory.

## Example 1. Coronahack

- Make a subdirectory named ``Coronahack``.
- Download the dataset from https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset 
- Store the dataset (i.e., "archive" directory) in the ``examples/datasets/RawData/Coronahack`` 

## Example 2. FEMNIST

- Make a subdirectory named ``FEMNIST``.
- Git clone https://github.com/TalwalkarLab/leaf.git
- In ``leaf/data/femnist``, do ``./preprocess.sh -s niid --sf 0.05 -k 0 -t sample`` which downloads **a small-sized dataset**.
- In a newly generated directory ``leaf/data/femnist/data``, copy the two directories ``train`` and ``test`` and paste them in ``examples/datasets/RawData/FEMNIST``   
 

## Example 3. CELEBA
- Make a subdirectory named ``CELEBA``.
- Download the dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 
- Generate preprocessed data according to the LEAF instruction: https://github.com/TalwalkarLab/leaf/tree/master/data/celeba
- Copy the two directories ``train`` , ``test`` and ``raw`` folders and paste in under ``examples/datasets/RawData/CELEBA``