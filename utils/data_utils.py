import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import cv2
from torch.utils import data
import shutil
import torch
from PIL import Image


def create_hdf5(data_dir, resize, dataset_name, type='.jpg'):
    labels = os.listdir('{}/training_set'.format(data_dir))
    labels.sort()
    labels_id = {}
    for k in range(len(labels)):
        labels_id[labels[k]] = k
    hdf5_folder = '{}/hdf5'.format(data_dir)
    if not os.path.exists(hdf5_folder):
        os.makedirs(hdf5_folder)
    for set in ['training', 'test']:
        addrs = {}
        path_to_data = '{}/{}_set'.format(data_dir, set)
        for label in labels:
            folder = '{}/{}'.format(path_to_data, label)
            addrs[label] = ['{}/{}'.format(folder, file) for file in os.listdir(folder) if type in file]
        hdf5_path = '{}/{}_{}.hdf5'.format(hdf5_folder, dataset_name, set)
        hdf5_file = h5py.File(hdf5_path, mode='w')

        nb_samples = sum(list(map(lambda x: len(x[1]), addrs.items())))
        hdf5_file.create_dataset('imgs', (nb_samples, resize, resize, 3), dtype=np.uint8)
        hdf5_file.create_dataset('labels_id', (nb_samples,), dtype=np.int)
        i = 0
        for label in labels:
            hdf5_file['labels_id'][i:i+len(addrs[label])] = labels_id[label]
            for img_file in addrs[label]:
                img = cv2.imread(img_file)
                img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hdf5_file['imgs'][i] = img
                i += 1
                if i % 1000 == 0:
                    print('Creation of {}.hdf5 / Set: {} / Iteration: [{}/{}] / Label: {}'.
                          format(dataset_name, set, i, nb_samples, label))
        hdf5_file.close()


class CustomDataset(data.Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.archive = h5py.File(hdf5_file, 'r')
        self.transform = transform
        self.data = list(map(lambda x: Image.fromarray(x), self.archive['imgs'][:]))
        self.labels_id = self.archive['labels_id'][:]
        self.classes = np.unique(self.labels_id)

    def __getitem__(self, index):
        data = self.data[index]
        label_id = self.labels_id[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label_id

    def __len__(self):
        return len(self.labels_id)

    def close(self):
        self.archive.close()


cifar_10 = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]}
cifar_100 = {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}
# cifar_10 = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}


def load_dataset(data_dir, resize, dataset_name, img_type):
    if dataset_name == 'cifar_10':
        mean = cifar_10['mean']
        std = cifar_10['std']
    elif dataset_name == 'cifar_100':
        mean = cifar_100['mean']
        std = cifar_100['std']
    else:
        print('Dataset not recognized. Data normalize with equal mean/std weights')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    hdf5_folder = '{}/hdf5'.format(data_dir)
    if os.path.exists(hdf5_folder):
        shutil.rmtree(hdf5_folder)
    create_hdf5(data_dir, resize, dataset_name, img_type)
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    hdf5_folder = '{}/hdf5'.format(data_dir)
    hdf5_train_path = '{}/{}_{}.hdf5'.format(hdf5_folder, dataset_name, 'training')
    hdf5_test_path = '{}/{}_{}.hdf5'.format(hdf5_folder, dataset_name, 'test')
    train_dataset = CustomDataset(hdf5_file=hdf5_train_path, transform=train_transform)
    val_dataset = CustomDataset(hdf5_file=hdf5_test_path, transform=val_transform)

    return [train_dataset, val_dataset]


def show_dataset(dataset, n=3, m=3):
    img = np.vstack((np.hstack((dataset[i][0].squeeze().permute(1, 2, 0).numpy() for _ in range(n)))
                   for i in range(m)))
    img -= np.min(img)
    img /= np.max(img)
    plt.imshow(img)
    plt.axis('off')


def get_dataloader(data_dir, resize, img_type, batch_size, num_workers):
    dataset_name = data_dir.split('/')[-1]
    train_data_set, val_data_set = load_dataset(data_dir, resize, dataset_name, img_type)
    train_dataloader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers)
    return [train_dataloader, val_dataloader]
