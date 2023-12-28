"""
This file is used to download the MNIST dataset from the web to a user-specified directory.
You need to provide the path to the directory where you want to save the dataset using --data_dir argument.
It is highly recommended to use absolute path for --data_dir argument, and use the same absolute path for data_dir argument in the client config file.
"""
import os
import argparse
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='Download MNIST dataset')
parser.add_argument('--data_dir', type=str, default='../../datasets', help='Directory to save MNIST dataset')
args = parser.parse_args()

if __name__ == '__main__':
    local_dir = os.path.join(args.data_dir, 'RawData')
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    datasets.MNIST(local_dir, download=True)