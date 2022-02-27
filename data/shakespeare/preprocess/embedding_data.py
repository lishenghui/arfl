import json
import os
import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

from language_utils import letter_to_indicate

embedding_dim = 8

def read_dir(data_dir):
    clients = []
    groups = []
    data = {}

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

    clients = list(sorted(data.keys()))
    return clients, groups, data


def embed_data(train_data_dir='../data/train', test_data_dir='../data/test', use_pca=True):
    train_clients, train_groups, raw_train_data = read_dir(train_data_dir)
    test_clients, test_groups, raw_test_data = read_dir(test_data_dir)

    chars = []
    for k in train_clients:
        for line in raw_train_data[k]['x']:
            for e in line:
                chars.append(e)
        for e in raw_train_data[k]['y']:
            chars.append(e)
    for k in test_clients:
        for line in raw_test_data[k]['x']:
            for e in line:
                chars.append(e)
        for e in raw_test_data[k]['y']:
            chars.append(e)

    chars = sorted(list(set(chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    embedding_vectors = {}
    embeddings_path = 'char_glove.txt'
    with open(embeddings_path, 'r') as f:
        for line in f:
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            char = line_split[0]
            embedding_vectors[char] = vec

    embedding_matrix = np.zeros((len(chars), 300))
    # embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
    for char, i in char_indices.items():
        # print ("{}, {}".format(char, i))
        embedding_vector = embedding_vectors.get(char)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    pca = PCA(n_components=embedding_dim)
    pca.fit(embedding_matrix)
    embedding_matrix_pca = np.array(pca.transform(embedding_matrix))
    print(embedding_matrix_pca)
    print(embedding_matrix_pca.shape)

    train_data = {}
    for k in list(train_clients):
        train_data[k] = {}
        train_data[k]['x'] = np.array([embed(line) for line in raw_train_data[k]['x']])
        train_data[k]['y'] = np.array([letter_to_indicate(e) for e in raw_train_data[k]['y']])

    test_data = {}
    for k in list(test_clients):
        test_data[k] = {}
        test_data[k]['x'] = [embed(line) for line in raw_test_data[k]['x']]
        test_data[k]['y'] = [letter_to_indicate(e) for e in raw_test_data[k]['y']]

    with open(os.path.join(os.path.split(train_data_dir)[0], 'data_cache.obj'), 'wb') as f:
        pickle.dump(train_clients, f)
        pickle.dump(train_data, f)
        pickle.dump(test_clients, f)
        pickle.dump(test_data, f)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

if __name__ == "__main__":
    embed_data('../data/train', '../data/test')
    # clients, groups, data = read_dir('../data/sampled_data')
    # chars = sorted(set([2,2,3]))