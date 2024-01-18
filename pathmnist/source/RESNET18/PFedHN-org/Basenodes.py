from dataset_main import gen_random_loaders
import torch

class Basenodes:
    def __init__(
            self,
            data_name,
            n_nodes,
            batch_size=64,
            classes_per_node = 2
    ):
        self.data_name = data_name
        self.n_nodes = n_nodes
        self.classes_per_node = classes_per_node
        self.batch_size = batch_size
        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
            self.data_name,
            self.n_nodes,
            self.batch_size,
            self.classes_per_node
        )

    def __len__(self):
        return self.n_nodes
    

# data = Basenodes("pathmnist", 50, 64, 2)


# for i in range(len(data.train_loaders)):
#     batch = next(iter(data.train_loaders[i]))
#     img, label = tuple(t for t in batch)
#     unique_classes = torch.unique(torch.tensor(label)).tolist()
#     print(f"The client id is {i} and the number of classes present in {i} is {unique_classes}")

# for i in range(len(data.train_loaders)):
#     batch = next(iter(data.test_loaders[i]))
#     img, label = tuple(t for t in batch)
#     unique_classes = torch.unique(torch.tensor(label)).tolist()
#     print(f"The client id is {i} and the number of classes present in {i} is {unique_classes}")