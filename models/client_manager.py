import importlib
import numpy as np
import os
import ray

from baseline_constants import MODEL_PARAMS
from client import Client
from utils.model_utils import read_data


class ClientManager():
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.dataset = args.dataset
        self.model = args.model
        self.clients = []

        self.model_path = '%s/%s.py' % (self.dataset, self.model)
        if not os.path.exists(self.model_path):
            print('Please specify a valid dataset and a valid model.')
        self.model_path = '%s.%s' % (self.dataset, self.model)
        print('############################## %s ##############################' % self.model_path)
        mod = importlib.import_module(self.model_path)
        self.ClientModel = getattr(mod, 'ClientModel')

        self.create_actors()

    def create_actors(self):
        # Create models
        model_params = MODEL_PARAMS[self.model_path]
        if self.args.lr != -1:
            model_params_list = list(model_params)
            model_params_list[0] = self.args.lr
            model_params = tuple(model_params_list)
        self.client_actors = [self.ClientModel.remote(self.seed, *model_params) for _ in range(self.args.num_actors)]

    def get_actors(self):
        return self.client_actors

    def set_model_params(self, params):
        ret = []
        for p in self.client_actors:
            r = p.set_params.remote(params)
            ret.append(r)
        ray.get(ret)

    def setup_clients(self, num_setup):
        """Instantiates clients based on given train and test data directories.

            Return:
                all_clients: list of Client objects.
            """
        eval_set = 'test'
        train_data_dir = os.path.join('..', 'data', self.dataset, 'data', 'train')
        test_data_dir = os.path.join('..', 'data', self.dataset, 'data', eval_set)

        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
        n = int(len(users) * num_setup)
        users = users[:n + 1]

        if len(groups) == 0:
            groups = [[] for _ in users]

        # Create models
        model_params = MODEL_PARAMS[self.model_path]
        if self.args.lr != -1:
            model_params_list = list(model_params)
            model_params_list[0] = self.args.lr
            model_params = tuple(model_params_list)

        for i, u in enumerate(users):
            # client = Client(u, train_data[u], test_data[u], self.seed )
            client = Client(u, train_data[u], test_data[u], self.seed + i)
            self.clients.append(client)

        return self.clients

    def corrupt_clients(self):
        # Randomly attack clients
        pc = self.args.pc
        ps = self.args.ps
        att_type = self.args.attack_type
        n = int(len(self.clients) * pc)
        np.random.seed(self.seed)
        selected_indexes = np.random.choice(range(len(self.clients)), n, replace=False)
        selected_indexes = np.sort(selected_indexes)
        for i in selected_indexes:
            self.clients[i].is_corrupted = True
            if att_type == 0:
                self.clients[i].poison_by_shuffling_labels(ps)
            elif att_type == 1:
                self.clients[i].poison_by_enforcing_label(ps, self.args)
            elif att_type == 2:
                self.clients[i].poison_by_noise(self.args)
            elif att_type == 3:  # clean training
                pass
        print("attacked clients: " + ','.join([str(i) for i in selected_indexes]))

    def get_clients_info(self):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """

        ids = [c.id for c in self.clients]
        # groups = {c.id: c.group for c in self.clients}
        num_train_samples = {c.id: c.num_train_samples for c in self.clients}
        num_test_samples = {c.id: c.num_test_samples for c in self.clients}
        return ids, None, num_train_samples, num_test_samples
