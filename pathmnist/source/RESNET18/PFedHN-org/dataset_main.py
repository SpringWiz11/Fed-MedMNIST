import random
from collections import defaultdict
import torch
import numpy as np
from dataset import get_data


def get_dataset(data_name):
    train_set, val_set, test_set = get_data(data_name=data_name)
    return train_set, val_set, test_set

def get_num_classes_samples(dataset):
    label_list = list()
    for i in range(len(dataset)):
        label_list.append(dataset[i][1])
        # print(dataset[i][0], " |||||| ", dataset[i][1])
        # break
    label_list = np.array(label_list)
    classes, num_samples = np.unique(label_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, label_list



def gen_classes_per_node(dataset, num_users, classes_per_user = 2, high_prob = 0.6, low_prob = 0.4):

    num_classes, num_samples,_ = get_num_classes_samples(dataset)
    count_per_class = (classes_per_user * num_users) // num_classes + 1 

    class_dict = {}

    for i in range(num_classes):
        probs = np.random.uniform(low_prob, high_prob, size = count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
    
    
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions

def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
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
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx




def gen_random_loaders(
    data_name,
    num_users,
    bz, 
    classes_per_user,      
):
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}
    dataloaders = []
    dataset = get_dataset(data_name)

    for i, d in enumerate(dataset):
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
            loader_params['shuffle'] = True
        
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))
    return dataloaders




# data = gen_random_loaders("pathmnist", 50, 64, 2)
# train, val, test = data

# for i in range(len(train)):
#     batch = next(iter(train[i]))
#     img, label = tuple(t for t in batch)
#     unique_classes = torch.unique(torch.tensor(label)).tolist()
#     print(f"The client id is {i} and the number of classes present in {i} is {unique_classes}")

# for i in range(len(test)):
#     batch = next(iter(test[i]))
#     img, label = tuple(t for t in batch)
#     unique_classes = torch.unique(torch.tensor(label)).tolist()
#     print(f"The client id is {i} and the number of classes present in {i} is {unique_classes}")



