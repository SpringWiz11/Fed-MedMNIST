from dataset_main import gen_random_loaders

class BaseNodes:
    def __init__(
            self,
            data_name,
            n_nodes,
            batch_size = 64,
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
            self.n_nodes,
            self.batch_size,
            self.classes_per_node
        )
        # self.train_loader, self.test_loader = get_train_test(self.data_name, self.node_id, self.batch_size)
    
    def __len__(self):
        return self.n_nodes
