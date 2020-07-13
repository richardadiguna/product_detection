import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



class ShopeeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        category = self.data.iloc[idx, 1]
        category_dir = '0{}'.format(category) if category < 10 else str(category)
        
        image_name = os.path.join(self.root_dir, category_dir, self.data.iloc[idx, 0])
        
        if self.test:
            image_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        
        image = Image.open(image_name).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'label': category}
    

def get_train_loader(csv_file, 
                     root_dir, 
                     batch_size, 
                     augment, 
                     random_seed, 
                     valid_size=0.1, 
                     shuffle=True,
                     num_workers=4,
                     pin_memory=True):
    
    normalize = transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
        
    valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
        
    train_dataset = ShopeeDataset(csv_file, root_dir, train_transform)
    valid_dataset = ShopeeDataset(csv_file, root_dir, valid_transform)
    
    dataset_size = len(train_dataset)
    indicies = list(range(dataset_size))
    split = int(np.floor(valid_size * dataset_size))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indicies)
    
    train_indicies, valid_indicies = indicies[split:], indicies[:split]
    
    train_sampler = SubsetRandomSampler(train_indicies)
    valid_sampler = SubsetRandomSampler(valid_indicies)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=pin_memory)
    
    valid_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=pin_memory)
    
    return train_loader, valid_loader


def get_test_loader(csv_file, 
                     root_dir, 
                     batch_size,
                     shuffle=False,
                     num_workers=4):
    
    normalize = transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))
    
    test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
    test_dataset = ShopeeDataset(csv_file, 
                                 root_dir, 
                                 test_transform, 
                                 test=True)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers)
    
    return test_loader
