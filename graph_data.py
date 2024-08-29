import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split

def split_data(data_dir, split):
    subdirs = os.listdir(data_dir)
    temp = []
    for subdir in subdirs:
        if subdir.endswith('landmarks'):
            temp.append(subdir)

    subdirs = temp

    images0 = []
    images1 = []
    images2 = []
    images3 = []
    images4 = []

    for subdir in subdirs:
        if subdir.startswith('stroke'):
            label = 1
        else:
            label = 0

        img_dict = {}
        img_list = os.listdir(data_dir + os.sep + subdir)
        
        for item in img_list:
            patient_id = item.split('_')
            if patient_id[0] not in img_dict.keys():
                img_dict[patient_id[0]] = []
            img_dict[patient_id[0]].append(data_dir + os.sep + subdir + os.sep + item)
        set_, set0 = train_test_split(list(img_dict.keys()), test_size=0.2, shuffle = True, random_state=3546)
        set_, set1 = train_test_split(set_, test_size=0.25, shuffle=True, random_state=45)
        set_, set2 = train_test_split(set_, test_size=0.33, shuffle=True, random_state=45)
        set3, set4 = train_test_split(set_, test_size=0.5, shuffle=True, random_state=45)

        for patient in set0:
            for k in range(0, len(img_dict[patient])):
                images0.append((img_dict[patient][k], label))
        for patient in set1:
            for k in range(0, len(img_dict[patient])):
                images1.append((img_dict[patient][k], label))
        for patient in set2:
            for k in range(0, len(img_dict[patient])):
                images2.append((img_dict[patient][k], label))   
        for patient in set3:
            for k in range(0, len(img_dict[patient])):
                images3.append((img_dict[patient][k], label))
        for patient in set4:
            for k in range(0, len(img_dict[patient])):
                images4.append((img_dict[patient][k], label))

        if split == 0:
            images_train = images0 + images1 + images2
            images_val = images3
            images_test = images4
        if split == 1:
            images_train = images1 + images2 + images3
            images_val = images4
            images_test = images0
        if split == 2:
            images_train = images2 + images3 + images4
            images_val = images0
            images_test = images1
        if split == 3:
            images_train = images3 + images4 + images0
            images_val = images1
            images_test = images2
        if split == 4:
            images_train = images4 + images0 + images1
            images_val = images2
            images_test = images3

    return images_train, images_val, images_test

def cross_data(data_dir):
    subdirs = os.listdir(data_dir)
    temp = []
    for subdir in subdirs:
        if subdir.endswith('landmarks'):
            temp.append(subdir)

    subdirs = temp

    images = []

    for subdir in subdirs:
        if subdir.startswith('stroke'):
            label = 1
        else:
            label = 0

        img_dict = {}
        img_list = os.listdir(data_dir + os.sep + subdir)
        
        for item in img_list:
            patient_id = item.split('_')
            if patient_id[0] not in img_dict.keys():
                img_dict[patient_id[0]] = []
            img_dict[patient_id[0]].append(data_dir + os.sep + subdir + os.sep + item)

        for patient in list(img_dict.keys()):
            for k in range(0, len(img_dict[patient])):
                images.append((img_dict[patient][k], label))

    return images


            
class GraphDataset(InMemoryDataset):
    def __init__(self, root, split=0, mode='train', transform=None, pre_transform=None, pre_filter=None, log=False):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.train_list, self.val_list, self.test_list = split_data(root, split)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'train':
            path, target = self.train_list[idx]
            file = np.load(path, allow_pickle=True)
            graph = Data(x=torch.tensor(file['l'], dtype=torch.float), edge_index=torch.tensor(file['e']).t().contiguous(), y=target)
            if self.transform is not None:
                graph = self.transform(graph)
            return graph
        elif self.mode == 'val':
            path, target = self.val_list[idx]
            file = np.load(path, allow_pickle=True)
            graph = Data(x=torch.tensor(file['l'], dtype=torch.float), edge_index=torch.tensor(file['e']).t().contiguous(), y=target)
            if self.transform is not None:
                graph = self.transform(graph)
            return graph
        elif self.mode == 'test':
            path, target = self.test_list[idx]
            file = np.load(path, allow_pickle=True)
            graph = Data(x=torch.tensor(file['l'], dtype=torch.float), edge_index=torch.tensor(file['e']).t().contiguous(), y=target)
            if self.transform is not None:
                graph = self.transform(graph)
            return graph
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        elif self.mode == 'val':
            return len(self.val_list)
        elif self.mode == 'test':
            return len(self.test_list)
        
class CrossDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, log=False):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.test_list = cross_data(root)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

    def __getitem__(self, idx):
        path, target = self.test_list[idx]
        file = np.load(path, allow_pickle=True)
        graph = Data(x=torch.tensor(file['l']), edge_index=torch.tensor(file['e']).t().contiguous(), y=target)
        if self.transform is not None:
            graph = self.transform(graph)
        return graph
        
    def __len__(self):
        return len(self.test_list)
        