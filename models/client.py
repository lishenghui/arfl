import numpy as np
import random
import ray

from utils.language_utils import ALL_LETTERS


class Client:

    def __init__(self, client_id, train_data={'x': [], 'y': []},
                 eval_data={'x': [], 'y': []}, seed=0,):
        self.id = client_id
        self.is_corrupted = False
        self.loss = {}
        self.train_data = train_data
        self.train_data['y'] = np.array(train_data['y'])
        self.eval_data = eval_data
        self.weight = 1.
        self.seed = seed

    def poison_by_shuffling_labels(self, p):  # Option 0
        """
            poison_by_shuffling_labels selects a fraction p of the samples and
            shuffles their labels randomly
        """
        sz = len(self.train_data['y'])
        n_poisoned = int(sz * p)
        poisoned_points = np.random.choice(sz, n_poisoned, replace=False)
        reordered = np.random.permutation(poisoned_points)
        self.train_data['y'][poisoned_points] = self.train_data['y'][reordered]

        return self.train_data

    def poison_by_enforcing_label(self, p, args):  # Option 1
        sz = len(self.train_data['y'])
        n_poisoned = int(sz * p)
        poisoned_points = np.random.choice(sz, n_poisoned, replace=False)
        if args.dataset == 'shakespeare':
            fake_label = random.randrange(len(ALL_LETTERS))
        elif args.dataset == 'femnist':
            fake_label = np.random.choice(range(62), 1, replace=False)
            # fake_label = 0
        elif args.dataset == 'sent140':
            l = np.random.choice([0, 1], 1, replace=False)
            fake_label = l
        # elif args.dataset == 'sent140':
        #     l = np.random.choice([0, 1], 1, replace=False)
        #     fake_label = l
        else:
            l = np.random.choice(range(10), 1, replace=False)
            fake_label = l
        self.train_data['y'][poisoned_points] = fake_label
        pass



    def poison_by_noise(self, args):

        x = np.array(self.train_data['x'])
        x_new = x
        if args.dataset == 'shakespeare':
            x_new = np.array([list(e) for e in x])
            l = len(x)
            for i in range(l):
                poisoned_points = np.random.choice(80, 40, replace=False)
                noise = np.random.choice(list(ALL_LETTERS), 40, replace=True)
                # reordered = np.random.permutation(poisoned_points)
                l = list(x_new[i])
                x_new[i] = np.array(l)
                x_new[i][poisoned_points] = noise
                # x_new[i][poisoned_points] = x_new[i][reordered]
            x_new = np.array([''.join(list(e)) for e in x_new])
        if args.dataset == 'femnist':
            scale = 0.7
            noise = np.random.normal(0, scale, x.shape)
            x_noisy = x + noise
            x_new = (x_noisy - np.min(x_noisy)) / (np.max(x_noisy) - np.min(x_noisy))

            # img = np.array(x_new[0]).reshape((28, 28))
            # plt.imshow(img, cmap='gray', aspect='equal')
            # plt.grid(False)
            # _ = plt.show()
        # modify client in-place
        self.train_data['x'] = x_new

    def train(self, num_epochs, batch_size, model, lr):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """

        data = self.train_data
        result = model.train.remote(data, num_epochs, batch_size, lr)

        # save_path = 'models'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # self.save_model(model, os.path.join(os.getcwd(), save_path, str(self.id) + '_' + str(self.is_corrupted)) )

        return result

    def test(self, set_to_use='test', model=None):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            step = 1
            # if len(self.train_data['y']) >= 200:
            #     step = 10
            data = {'x': self.train_data['x'][::step], 'y': self.train_data['y'][::step]}
        else:
            data = self.eval_data
        results = model.test.remote(data)
        return results

    def get_train_loss(self):
        return self.loss['train']

    # def get_params(self):
    #     return self.model.get_params()
    #
    #
    # def set_params(self, params):
    #     self.model.set_params(params)

    def set_weight(self, w):
        self.weight = w

    def save_model(self, model, path):
        results_id = model.save_model.remote(path)
        ray.get(results_id)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

