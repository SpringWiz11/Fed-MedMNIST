import torchvision.transforms as transforms
import torch.utils.data as data
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader


def get_dataset(data_name: str, Batch_size: int = 128):
    download = True
    info = INFO[data_name]
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load the dataset

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform = data_transform, download = download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    #encapsulate the dataset into a class

    train_loader = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True)
    train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=False)
    val_loader = DataLoader(dataset = val_dataset, batch_size = Batch_size, shuffle = False)

    return train_loader, train_loader_at_eval, test_loader, val_loader

x, y, z, a = get_dataset('pathmnist')
print(x)