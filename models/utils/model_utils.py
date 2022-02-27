import json
import numpy as np
import os
import pickle
from collections import defaultdict
from scipy.sparse import csr_matrix


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)

    if not isinstance(data_x, csr_matrix):
        np.random.shuffle(data_x)
        np.random.shuffle(data_y)
        sz = len(data_x)
    else:
        sz = data_x.shape[0]
    # loop through mini-batches
    for i in range(0, sz, batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    pre_dir = os.path.split(train_data_dir)[0]
    cache_path = os.path.join(pre_dir, "data_cache.obj")
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
        train_groups = []
    else:
        train_clients, train_groups, train_data = read_dir(train_data_dir)
        test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert sorted(train_clients) == sorted(test_clients)
    # assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
