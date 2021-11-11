## Dataset from Torchvision
Example. MNIST
- In ```appfl/config/config.yaml```, by choosing ```dataset: mnist```, ```appfl/config/dataset/mnist.yaml``` is equipped.
- If there is no MNIST dataset in the ```appfl/datasets``` directory, APPFL will download the dataset from torchvision and store it in this directory. 

## Coronahack-Chest-XRay  
- ```mkdir Coronahack``` in this directory
- Download the dataset from https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset
- Place the dataset (i.e., "archive" directory) to the ```appfl/datasets/Coronahack``` directory
- In ```appfl/config/config.yaml```, by choosing ```dataset: coronahack```, ```appfl/config/dataset/coronahack.yaml``` is equipped.

## FEMNIST 
- ```mkdir FEMNIST``` in this directory
- ```Git clone https://github.com/TalwalkarLab/leaf.git```
- Follow the instruction in https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
- As an example, in ```leaf/data/femnist```, do ```./preprocess.sh -s niid --sf 0.05 -k 0 -t sample``` which downloads a small-sized dataset (#training data=36708, #features=784, #classes=62, #clients=195)
- In a newly generated directory ```leaf/data/femnist/data```, copy the two directories ```train``` and ```test``` and paste them in ```appfl/datasets/FEMNIST```
- In ```appfl/config/config.yaml```, by choosing ```dataset: femnist```, ```appfl/config/dataset/femnist.yaml``` is equipped.
