## Dataset from torchvision
Example. MNIST
- In ```appfl/config/config.yaml```, by choosing ```dataset: mnist```, ```appfl/config/dataset/mnist.yaml``` is equipped.
- If there is no MNIST dataset in the ```appfl/datasets``` directory, APPFL will download the dataset from torchvision and store it in this directory. 

## CoronaHack-Chest Dataset 
  1. Download the dataset from https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset
  2. Place the dataset (i.e., "archive" directory) to the ```appfl/datasets``` directory
  3. In ```appfl/config/config.yaml```, by choosing ```dataset: covid```, ```appfl/config/dataset/covid.yaml``` is equipped.

## FEMNIST

