import torchvision
from torchvision.transforms import ToTensor

import pickle

def write_data():

    # test data for a server
    test_data_raw = torchvision.datasets.MNIST(
        f"./_data", download=True, train=False, transform=ToTensor()
    )

    test_data = {"x": [], "y": []}
    for idx in range(len(test_data_raw)):
        test_data["x"].append(test_data_raw[idx][0].tolist())
        test_data["y"].append(test_data_raw[idx][1])

    with open('mnist_test_data.pickle', 'wb') as f:
        pickle.dump(test_data, f)
    

if __name__ == "__main__":
    write_data()
