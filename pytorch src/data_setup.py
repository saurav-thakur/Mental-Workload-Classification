import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from data_collection import collect_data
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self,data,labels=None,transforms=None,dataset_type=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms
        self.dataset_type = dataset_type


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]

        if self.dataset_type == "train":
            label = self.labels[index]

        if self.transforms:
            image = self.transforms(image)
        if self.dataset_type == "train":
            return image,label
        return image
    

def prepare_and_split_data(dataset_path,dataset_type):


    if dataset_type == "train":
        data, labels = collect_data(path=dataset_path,dataset_type=dataset_type)

        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

        dataset = CustomDataset(data=data,labels=labels)

        return dataset.data,dataset.labels

    elif dataset_type == "test":

        data = collect_data(path=dataset_path,dataset_type=dataset_type)
        data = torch.from_numpy(data)

        dataset = CustomDataset(data=data,dataset_type=dataset_type)

        return dataset






