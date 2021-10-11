import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MnistLocalDataset(Dataset):
    def __init__(self, images, labels, client_id):
        self.images = images
        self.labels = labels.astype(int)
        self.client_id = client_id
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index].reshape(28, 28), mode='L')
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)