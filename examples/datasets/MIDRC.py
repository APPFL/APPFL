import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import PIL.Image as Image
from imgaug import augmenters as iaa
import matplotlib.pylab as plt
import csv
import time
import os
from random import shuffle

# race2idx = {'White':0, 'Black or African American':1, 'Asian':2, 'Other':3}
race2idx = {'White':0, 'Black or African American':1, 'Other':2}
sex2idx = {'Male':0, 'Female':1}
states2idx = {'IL':0, 'NC':1, 'CA':2, 'IN':3, 'TX':4}

class MidrcDataset(Dataset):
    """Midrc dataset."""

    def __init__(self, csv_path, base_path, augment_times=1, transform=None, n_samples=None):
        """
        Args:        
            csv_path (string): The csv which contains the info we needed.
            augment_times (int): how many times we want to augment the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = []
        self.labels = []
        self.transform = transform
        with open(csv_path, 'r') as csv_file:
            rows = csv_file.read().split('\n')[1:]
            count = 0
            shuffle(rows) 
            for row in rows:
                if row:
                    if n_samples and count >= n_samples:
                        break
                    if count % 1000 == 0:
                        print(count)
                    row = row.split(',')
                    is_covid = int(row[2] == 'Yes')
                    img_path = os.path.join(base_path, 'states', row[0].split('/')[-1])
                    img_origin = Image.open(img_path).convert('RGB')
                    count += 1
                    # if sex == 'Male':
                    # if race == 'Black or African American':
                    # if race == 'White':
                    for _ in range(augment_times):
                        if self.transform:
                            img = self.transform(img_origin)
                        self.labels.append(is_covid)
                        self.imgs.append(img)
                        
        print(len(self.labels))
        csv_file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

class MidrcMLTDataset(Dataset):
    """Midrc Multi-task learning with demographic info dataset."""

    def __init__(self, csv_path, base_path, augment_times=1, transform=None, n_samples=None):
        """
        Args:        
            csv_path (string): The csv which contains the info we needed.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = []
        self.labels = []
        self.races = []
        self.genders = []
        self.ages = []
        self.states = []
        self.transform = transform
        with open(csv_path, 'r') as csv_file:
            rows = csv_file.read().split('\n')[1:]
            count = 0
            shuffle(rows)
            for row in rows:
                if row:
                    if n_samples and count >= n_samples:
                        break
                    if count % 1000 == 0:
                        print(count)
                    img_path, race, is_covid, sex, age, state = row.split(',')
                    img_path = os.path.join(base_path, 'states', img_path.split('/')[-1])
                    img_origin = Image.open(img_path).convert('RGB')
                    race = race2idx[race] if race in race2idx else race2idx['Other']
                    count += 1
                    for _ in range(augment_times):
                        if self.transform:
                            img = self.transform(img_origin)
                        self.labels.append(int(is_covid == 'Yes'))
                        self.imgs.append(img)
                        self.races.append(race)
                        self.genders.append(sex2idx[sex])
                        self.ages.append(int(age)//10)
                        self.states.append(states2idx[state])
        csv_file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):   
        return {'img': self.imgs[idx], 'targets': (self.labels[idx], self.races[idx], self.genders[idx], self.ages[idx])}
    
    def get_labels(self):
        return self.races

if __name__ == '__main__':
    csv_file = '/u/enyij2/data/midrc/meta_info/MIDRC_table_CA_test.csv'
    base_path = '/u/enyij2/data/midrc/cr_states/'
    transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,)),
			]
		)
    test = MidrcDataset(csv_file, base_path, transform=transform)
    print(test.labels)
    for i in range (0, len(test), 100): 
        print(test[i][0].shape)
        print(test[i][0].max(), test[i][0].min())
        plt.imshow(test[i][0][0,:,:])
        plt.savefig('test_img.png')
        break

    loader = torch.utils.data.DataLoader(test, batch_size=16)