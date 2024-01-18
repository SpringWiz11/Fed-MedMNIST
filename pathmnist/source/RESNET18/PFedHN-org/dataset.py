import medmnist
import torchvision.transforms as transforms
from medmnist import INFO

def get_data(data_name):

    data_flag = data_name
    download = True
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    val_dataset = DataClass(split='val',transform=data_transform, download=download)

    return train_dataset, val_dataset, test_dataset