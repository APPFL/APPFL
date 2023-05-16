import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def test_transform(dataset):
    """Return the test transformation for different datast (MNIST/CIFAR10)"""
    if dataset == "MNIST":
        return transforms.ToTensor()
    elif dataset == "CIFAR10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError
    
def train_transform(dataset):
    """Return the train transformation for different datast (MNIST/CIFAR10)"""
    if dataset == "MNIST":
        return transforms.ToTensor()
    elif dataset == "CIFAR10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),                        
            normalize,
        ])
    else:
        raise NotImplementedError

def plot(K, classes, res, filename):
    """Visualize the data distribution for different classes"""
    _, ax = plt.subplots(figsize=(20, K/2))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', 
            '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', 
            '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', 
            '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    ax.barh(range(K), res[0], color=colors[0])
    for i in range(1, len(classes)):
        ax.barh(range(K), res[i], left=np.sum(res[:i], axis=0), color=colors[i])

    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename)
