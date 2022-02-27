import numpy as np
import ray
import sys
from model import FedModel
from utils.language_utils import letter_to_indicate, word_to_indices

sys.path.append("..")
from utils.args import GPU_PER_ACTOR


@ray.remote(num_gpus=GPU_PER_ACTOR)
class ClientModel(FedModel):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden, test_batch_size):
        FedModel.__init__(self, seed, lr, test_batch_size)
        self.lr = lr
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.net = self.create_model()

    def create_model(self, lr=None):
        if lr == None:
            lr = self.lr
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Dropout, Dense, LSTM
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


        inputs = Input([self.seq_len], dtype='int32')

        # x = Embedding(self.num_classes, 8, input_length=self.seq_len)(inputs)

        class CustomEmbedding(tf.keras.layers.Layer):
            def __init__(self, input_dim, output_dim, **kwargs):
                super(CustomEmbedding, self).__init__(**kwargs)
                self.input_dim = input_dim
                self.output_dim = output_dim

            def build(self, input_shape):
                self.embeddings = self.add_weight(name='embedding', shape=(self.input_dim, self.output_dim),
                                                    initializer='random_normal', dtype='float32')

            def call(self, inputs):
                return tf.nn.embedding_lookup(self.embeddings, inputs)

        Embedding = CustomEmbedding(self.num_classes, 8)
        x = Embedding(inputs)

        x = LSTM(self.n_hidden, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(self.n_hidden, )(x)
        x = Dropout(0.2)(x)


        outputs = Dense(units=self.num_classes)(x)
        model = Model(inputs=inputs, outputs=outputs)

        self.opt = tf.keras.optimizers.SGD(learning_rate=lr, clipvalue=1.0)
        print('learning rate:', lr)

        # def loss(labels, pred):
        #     loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels)
        #     return tf.reduce_mean(loss)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=self.opt, loss=loss, metrics=['accuracy'])
        print(model.summary())
        return model

    def process_x(self, raw_x_batch):

        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch, dtype=np.int32)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_indicate(c) for c in raw_y_batch]
        y_batch = np.array(y_batch, dtype=np.int32)
        return y_batch


