import numpy as np
import ray
import sys
from model import FedModel

sys.path.append("..")
from utils.args import GPU_PER_ACTOR

IMAGE_SIZE = 28


@ray.remote(num_gpus=GPU_PER_ACTOR)
class ClientModel(FedModel):
    def __init__(self, seed, lr, num_classes, test_batch_size):
        FedModel.__init__(self, seed, lr, test_batch_size)
        self.num_classes = num_classes
        self.net = self.create_model()

    def create_model(self, lr=None):
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Input, Conv2D, Reshape, MaxPool2D, Flatten
        from tensorflow.keras import Model

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

        """Model function for CNN."""
        inputs = Input(dtype=tf.float32, shape=(IMAGE_SIZE * IMAGE_SIZE,), name="Input")

        x = Reshape([IMAGE_SIZE, IMAGE_SIZE, 1])(inputs)
        x = Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)(x)
        x = MaxPool2D(pool_size=[2, 2], strides=2)(x)
        x = Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)(x)
        x = MaxPool2D(pool_size=[2, 2], strides=2)(x)
        x = Flatten()(x)
        x = Dense(units=128, activation=tf.nn.relu)(x)
        outputs = Dense(units=self.num_classes)(x)

        model = Model(inputs=inputs, outputs=outputs)
        print(model.summary())
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="SGD", loss=loss, metrics=['accuracy'])
        return model

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
