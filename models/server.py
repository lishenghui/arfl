import logging
import numpy as np
import ray
import time

from utils.algorithms import cosine_similarity_lists, bi_partitioning
from utils.args import ARGS
from utils.args import LOG_FILENAME


class Server:
    def __init__(self, clients, manager, args):
        self.model = ray.get(manager.get_actors()[0].get_params.remote())
        self.selected_clients = []
        self.selected_indexes = []
        self.updates = []
        self.args = args
        self.seed = args.seed
        self.clients = np.array(clients)
        self.num_clients = len(self.clients)
        self.data_sizes = np.array([len(c.train_data['y']) for c in self.clients])
        self.weights = np.ones(shape=(self.num_clients))
        self.client_manager = manager
        self.total_num_samples = sum([c.num_train_samples for c in self.clients])
        self.lr = ARGS.lr
        self.reg_weight = self.total_num_samples if args.reg_weight is None else args.reg_weight * self.total_num_samples
        print('Setting lambda as ', self.reg_weight)
        
        # for afa
        self.alpha_0 = 3
        self.beta_0 = 3
        self.prob = np.ones_like(self.clients) / 2
        self.num_good = np.zeros_like(self.clients)
        self.num_bad = np.zeros_like(self.clients)
        self.similarities = np.zeros_like(self.clients)

        logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format='%(message)s')

    def update_alpha(self, round_num):
        self.test_model(round_num, clients_to_test=self.selected_clients, set_to_use='train', log=False)  # Evaluate training loss
        idxs = [x for x, _ in sorted(enumerate(self.clients), key = lambda x: x[1].get_train_loss())]
        eta_optimal =  self.clients[idxs[0]].get_train_loss()  + self.reg_weight / self.clients[idxs[0]].num_train_samples
        for p in range(0, len(idxs)):
            eta = (sum([self.clients[i].num_train_samples * self.clients[i].get_train_loss() for i in idxs[:p+1]]) + self.reg_weight ) \
                    / sum([self.clients[i].num_train_samples for i in idxs[:p+1]])
            if eta - self.clients[idxs[p]].get_train_loss() < 0:
                break
            else:
                eta_optimal = eta
        weights = [c.num_train_samples * max( eta_optimal - c.get_train_loss(), 0) / self.reg_weight for c in self.clients]
        for i, c in enumerate(self.clients):
            w = c.num_train_samples * max( eta_optimal - c.get_train_loss(), 0) / self.reg_weight
            c.set_weight(w)
            self.weights[i] = w
        return weights, np.dot(weights, [c.get_train_loss() for c in self.clients]) + \
                self.reg_weight * np.sum([w ** 2 / c.num_train_samples for w, c in zip(weights, self.clients)]) / 2

    def lr_schedule_v0(self, round):
        if self.lr is None:
            self.lr = ARGS.lr
    
        if round != 0 and ARGS.decay:
            self.lr = self.lr * 0.99
    
        return self.lr

    def lr_scheduler(self, epoch):
        if not ARGS.decay:
            return self.lr
        new_lr = self.lr
        if epoch <= 300:
            pass
        elif epoch > 301 and epoch <= 500:
            new_lr = self.lr * 0.1
        else:
            new_lr = self.lr * 0.01
        print('new lr:%.2e' % new_lr)
        return new_lr
    
    def select_clients(self, my_round, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        possible_clients = self.clients
        num_clients = min(num_clients, len(possible_clients))

        possible_clients = np.array(possible_clients)
        # non_zeros = len([e for e in self.weights if e > 0])
        # num_clients = min(non_zeros, num_clients)
        # candidates = [i for i in range(len(possible_clients)) if self.weights[i] > 0]
        np.random.seed(self.seed * 1000 + my_round)
        candidates = [i for i in range(len(possible_clients))]
        while True:
            selected_index = np.random.choice(candidates, num_clients, replace=False)
            if sum([possible_clients[i].weight for i in selected_index]) != 0:
                break
        self.selected_indexes = selected_index
        
        print(self.selected_indexes)
        self.selected_clients = possible_clients[selected_index]
        
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs, batch_size, round):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        clients = self.selected_clients
        num_clients = len(clients)

        all_results = []
        actors = self.client_manager.get_actors()
        actor_size = len(actors)
        result_ids = []
        t1 = time.time()

        self.client_manager.set_model_params(self.model)
        lr = self.lr_scheduler(round)
        for i, client in enumerate(clients):
            results_id = client.train(num_epochs, batch_size, actors[i % actor_size], lr)
            result_ids.append(results_id)
            if (i + 1) % actor_size == 0 or i + 1 == num_clients:
                results = ray.get(result_ids)
                result_ids = []
                all_results.extend(results)
                self.client_manager.set_model_params(self.model)

        weights = [c.weight for c in clients]
        samples = [c.num_train_samples for c in clients]
        self.updates.extend((n, u, w) for n, (c, u), w in zip(samples, all_results, weights))
        print('training time {}'.format(time.time() - t1))

    def aggregate(self, method='arfl'):
        self.__getattribute__('aggregate_' + method)()

    def aggregate_fedavg(self):
        # total_weight = 0.
        # base = [0] * len(self.updates[0][1])
        # for (client_samples, client_model, _) in self.updates:
        #     total_weight += client_samples
        #     for i, v in enumerate(client_model):
        #         base[i] += (client_samples * v)
        # averaged_soln = [v / total_weight for v in base]
        #
        # self.model = averaged_soln
        # self.updates = []


        base = [0] * len(self.updates[0][1])
        weights = [w for (_, _, w) in self.updates]
        nor_weights = np.array(weights) / np.sum(weights)
        
        for idx, (client_samples, client_model, _) in enumerate(self.updates):
            for i, v in enumerate(client_model):
                base[i] += (nor_weights[idx] * v.astype(np.float64))
        self.model = base
        self.updates = []
        
    def aggregate_arfl(self):
        if sum([c.weight for c in self.selected_clients]) >= 0:
            base = [0] * len(self.updates[0][1])
            weights = [w for (_, _, w) in self.updates]
            nor_weights = np.array(weights) / np.sum(weights)
    
            for idx, (client_samples, client_model, _) in enumerate(self.updates):
                for i, v in enumerate(client_model):
                    base[i] += (nor_weights[idx] * v.astype(np.float64))
            self.model = base
        self.updates = []


    def aggregate_rfa(self, maxiter=4, eps=1e-5, ftol=1e-6, alphas=None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        points = [client_model for (_, client_model, _) in self.updates]
        if alphas is None:
            alphas = [w for (_, _, w) in self.updates]
        alphas = np.asarray(alphas, dtype=points[0][0].dtype) / sum(alphas)
        median = self.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1
        obj_val = self.geometric_median_objective(median, points, alphas)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, points, alphas)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        self.model = median
        self.updates = []

    def aggregate_cfl(self, thresh=0.02):
        l = len(self.updates)
        if l > 1:
            diffs = []
            for update in self.updates:
                diff = np.array([])
                for i, e in enumerate(update[1]):
                    diff = np.append(diff, (e - self.model[i]).flatten())
                diffs.append(diff)

            sim = np.zeros(shape=(l, l))
            for i in range(0, l):
                for j in range(i + 1, l):
                    sim[i][j] = cosine_similarity_lists(diffs[i], diffs[j])

            sim = sim.T + sim #+ np.eye(l)

            clusters = bi_partitioning(sim)

            sim_cross = -1
            for i in clusters[0]:
                for j in clusters[1]:
                    if sim[i][j] > sim_cross:
                        value = sim[i][j]
                        sim_cross = value

            bad_index = []
            if sim_cross >= thresh:
                good_index = clusters[0] + clusters[1]
            elif len(clusters[0]) >  len(clusters[1]):
                good_index = clusters[0]
                bad_index = clusters[1]
            else:
                good_index = clusters[1]
                bad_index = clusters[0]

            for e in bad_index:
                self.weights[self.selected_indexes[e]] = 0.0
            self.updates = [e for i, e in enumerate(self.updates) if i in good_index]
        self.aggregate_fedavg()

    def aggregate_mkrum(self):
        diffs = []
        for update in self.updates:
            diff = np.array([])
            for i, e in enumerate(update[1]):
                diff = np.append(diff, (e - self.model[i]).flatten())
            diffs.append(diff)

        l = len(self.updates)
        q = int( l // 2)

        dis = np.zeros(shape=(l, l))
        for i in range(0, l):
            for j in range(i + 1, l):
                dis[i][j] = self.l2dist([diffs[i]], [diffs[j]])
                # dis[i][j] = 1 - cosine_similarity_lists(diffs[i], diffs[j])

        dis = dis.T + dis
        krums = [np.sum(sorted(d)[:l - q - 2]) for d in dis]
        idxs = np.argsort(krums)[ : l // 2]

        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for i in idxs:
            client_samples, client_model, _ = self.updates[i]
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v)
        averaged_soln = [v / total_weight for v in base]
        print('using client:', idxs)
        self.model = averaged_soln
        self.updates = []


    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = np.sum(weights)
        weighted_updates = [np.zeros_like(v) for v in points[0]]

        for w, p in zip(weights, points):
            for j, weighted_val in enumerate(weighted_updates):
                weighted_val += (w / tot_weights) * p[j]

        return weighted_updates

    def test_model(self, round_number, clients_to_test=None, set_to_use='test', log=True):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.clients

        num_clients = len(clients_to_test)
        self.client_manager.set_model_params(self.model)
        actors = self.client_manager.get_actors()
        actor_size = len(actors)
        result_ids = []
        all_results = []
        for i, client in enumerate(clients_to_test):
            results_id = client.test(set_to_use, actors[i % actor_size])
            result_ids.append(results_id)
            if (i + 1) % actor_size == 0 or i + 1 == num_clients:
                results = ray.get(result_ids)
                result_ids = []
                all_results.extend(results)

        for client, result in zip(clients_to_test, all_results):
            c_metrics = result
            metrics[client.id] = c_metrics
            client.loss[set_to_use] = result['loss']
            if log:
                if set_to_use=='test':
                    num_samples = client.num_test_samples
                else:
                    num_samples = client.num_train_samples
                info = [client.id, round_number + 1, num_samples, set_to_use, result["loss"], result['accuracy'],
                        client.weight, client.is_corrupted]
                log_info = ','.join([str(i) for i in info])
                logging.info(log_info)

        return metrics

    def get_clients_losses(self, clients):
        losses = [client.get_train_loss() for client in clients]
        return losses

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def set_model(self, model):
        self.model = model
        
    def save_model(self, path):
        model = self.client_manager.get_actors()[0]
        self.clients[0].save_model(model, path)
