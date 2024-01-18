import medmnist
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from medmnist import INFO, Evaluator


#print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
def get_path_mnist(): 
    data_flag = "pathmnist"
    info = INFO[data_flag]
    download = True
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    val_dataset = DataClass(split='val',transform=data_transform, download=download)

    return train_dataset, val_dataset, test_dataset

def get_info():
    info = INFO['pathmnist']
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    return task, n_channels, n_classes

# x, y, z = get_path_mnist()
# img, label = x[1]
# print(type(img))
# print(type(label))

# print("The length of the train set = ",len(x))
# print("The length of the val set",len(y))
# print("The length of the test set",len(z))

# print(len(x[0]))
# print(len(y[0]))
# print(len(z[0]))

