import json
import os
import sys

import numpy as np
import tensorflow_federated as tff

from embedding_data import embed_data

sys.setrecursionlimit(30000)

SEQ_LENGTH = 80
MIN_LENGTH = 80
NUM_MIN_SAMPLE = 100
STEP = 5
utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)


train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

def split_word(word, texts, labels):
    print(word)
    if len(word) < MIN_LENGTH:
        return

    if len(word) < SEQ_LENGTH + 1:
        x = word[:-1]
        y = word[-1]
        word = ""
    else:
        x = word[:SEQ_LENGTH]
        y = word[SEQ_LENGTH]
        word = word[STEP:]
    texts.append(x)
    labels.append(y)
    split_word(word, texts, labels)

def split_snippet(snippets):
    x, y = [], []
    for word in snippets:
        word = ' '.join(word.split())
        split_word(word, x, y)
    return x, y

def convert_dataset(dataset):
    ids = dataset.client_ids
    user_data = {}
    for id in ids:
        raw_example_dataset = dataset.create_tf_dataset_for_client(id)
        data = list(raw_example_dataset)
        snippets = [e['snippets'].numpy().decode('utf-8') for e in data]

        x, y = split_snippet(snippets)
        user = {'x' : x, 'y' : y}
        user_data[id] = user

    return user_data

train_set = convert_dataset(train_data)
test_set = convert_dataset(test_data)
keys = list(train_set.keys())
for id in keys:
    if len(test_set[id]['y']) == 0 or len(train_set[id]['y']) < NUM_MIN_SAMPLE:
        train_set.pop(id)
        test_set.pop(id)

train_data = {"users": list(train_set.keys()), 'num_samples': [len(train_set[k]['y']) for k in train_set.keys()],
              'user_data': train_set}
test_data = {"users": list(test_set.keys()), 'num_samples': [len(test_set[k]['y']) for k in test_set.keys()],
             'user_data': test_set}
num_train_samples = np.sum(train_data['num_samples'])
print(num_train_samples)
os.system('mkdir -p ../data/train')
with open('../data/train/train_data.json', 'w') as f:
    json.dump(train_data, f)

os.system('mkdir -p ../data/test')
with open('../data/test/test_data.json', 'w') as f:
    json.dump(test_data, f)



embed_data()
