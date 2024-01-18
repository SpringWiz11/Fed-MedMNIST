import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
from dataset import get_path_mnist, get_info
from collections import defaultdict
import random
import os
from torch.utils.data import DataLoader

def get_datasets():
    x, y, z = get_path_mnist()
    return x, y, z
    

def get_num_classes_samples(dataset):
    
    task, n_channels, n_classes = get_info()
    label_list = list()
    for i in range(len(dataset)):
        label_list.append(dataset[i][1])
    label_list = np.array(label_list)
    classes, num_samples = np.unique(label_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, label_list

def gen_classes_per_node(
        dataset,
        num_users,
        classes_per_user = 2,
        high_prob = 0.6,
        low_prob = 0.4,
):
    x, y, z = get_num_classes_samples(dataset)

    count_per_class = (classes_per_user * num_users) // x + 1
    class_dict = {}
    for i in range(x):
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {"count": count_per_class, "prob": probs_norm}
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]["count"] for i in range(x)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]["count"] -= 1
        class_partitions["class"].append(c)
        class_partitions["prob"].append([class_dict[i]["prob"].pop() for i in c])
    return class_partitions

def gen_data_split(dataset, num_users, class_partitions):
    """Divide data indexes for each client based on class_partition.

    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(
            class_partitions["class"][usr_i], class_partitions["prob"][usr_i]
        ):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx



def gen_random_loaders(num_users, bz, classes_per_users):
    loader_params = {
        "batch_size": bz,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": 0,
    }

    data_loaders = []
    datasets = get_datasets()
    for i, d in enumerate(datasets):
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_users)
            loader_params["shuffle"] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        # create subsets for each client
        subsets = [torch.utils.data.Subset(d, x) for x in usr_subset_idx]
        # create dataloaders from subsets
        data_loaders.append(
            [torch.utils.data.DataLoader(x, **loader_params) for x in subsets]
        )
    return data_loaders

# def read_data(dataset, idx, is_train=True):
#     if is_train:
#         train_data_dir = os.path.join(dataset, 'train/')

#         train_file = train_data_dir + str(idx) + '.npz'
#         with open(train_file, 'rb') as f:
#             train_data = np.load(f, allow_pickle=True)['data'].tolist()

#         return train_data

#     else:
#         test_data_dir = os.path.join(dataset, 'test/')

#         test_file = test_data_dir + str(idx) + '.npz'
#         with open(test_file, 'rb') as f:
#             test_data = np.load(f, allow_pickle=True)['data'].tolist()

#         return test_data

# def read_client_data(dataset, idx, is_train=True):

#     if is_train:
#         train_data = read_data(dataset, idx, is_train)
#         X_train = torch.Tensor(train_data['x']).type(torch.float32)
#         y_train = torch.Tensor(train_data['y']).type(torch.int64)

#         train_data = [(x, y) for x, y in zip(X_train, y_train)]
#         return train_data
#     else:
#         test_data = read_data(dataset, idx, is_train)
#         X_test = torch.Tensor(test_data['x']).type(torch.float32)
#         y_test = torch.Tensor(test_data['y']).type(torch.int64)
#         test_data = [(x, y) for x, y in zip(X_test, y_test)]
#         return test_data

# def load_train_data(dataset, id, batch_size=None):
#     if batch_size == None:
#         batch_size = batch_size
#     train_data = read_client_data(dataset, id, is_train=True)
#     return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

# def load_test_data(dataset, id, batch_size=None):
#     if batch_size == None:
#         batch_size = batch_size
#     test_data = read_client_data(dataset, id, is_train=False)
#     return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)


# def get_train_test(dataset, id, batch_size):
#     train_loader = load_train_data(dataset,id, batch_size)
#     test_loader = load_test_data(dataset,id, batch_size)
#     return train_loader, test_loader