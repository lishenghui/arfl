"""Interfaces for ClientModel and ServerModel."""
import gc
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras import backend as K

from baseline_constants import ACCURACY_KEY


class FedModel(ABC):
    def __init__(self, seed, lr, test_batch_size=64):
        super(FedModel, self).__init__()
        self.lr = lr
        self.seed = seed
        self.datagen = None
        self.test_bs = test_batch_size
        tf.random.set_seed(123 + self.seed)
        self.net = None

        self.saver = tf.compat.v1.train.Saver

        # TODO: get graph size
        self.size = 0
        # stats = FlopCoKeras(self.net)
        self.flops = 0
        # self.flops = stats.total_flops
        np.random.seed(self.seed)

    def set_params(self, model_params):
        self.net.set_weights(model_params)
        return True

    def get_params(self):
        return self.net.get_weights()

    def save_model(self, path):
        self.net.save(path)
        return True

    def create_model(self, lr):
        pass

    def get_train_data(self, data, labels):
        gen = self.datagen.flow(data, labels, batch_size=len(labels))
        x, y = next(gen)
        return x, y

    def train(self, data, num_epochs, batch_size, lr):
        features = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        if lr != self.lr:
            K.set_value(self.net.optimizer.lr, lr)
            self.lr = lr

        if self.datagen:
            x, y = self.get_train_data(features, labels)
            self.net.fit(x, y, verbose=0, epochs=num_epochs)
            gc.collect()
        else:
            self.net.fit(features, labels, batch_size=batch_size, verbose=0, epochs=num_epochs)

        update = self.get_params()
        return 0, update

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        features = self.process_x(data['x'])
        labels = self.process_y(data['y'])


        loss, acc = self.net.evaluate(features, labels, verbose=0, batch_size=self.test_bs)

        return {ACCURACY_KEY: acc, 'loss': loss}

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass
