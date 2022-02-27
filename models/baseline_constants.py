"""Configuration file for common models/experiments"""

MAIN_PARAMS = {
    'sent140': {
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (3000, 5, 40)
    },
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (2000, 20, 20)
    },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 5, 2)
    },
    'celeba': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
    },
    'synthetic': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (3000, 5, 5)
    },
    'cifar10': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (2000, 20, 20)
    },
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'sent140.bag_dnn': (0.0003, 2),  # lr, num_classes
    'sent140.stacked_lstm': (0.0003, 25, 2, 100),  # lr, seq_len, num_classes, num_hidden
    'sent140.bag_log_reg': (0.0003, 2),  # lr, num_classes
    'sent140.log_reg': (0.3, 2, 400000),  # lr, num_classes
    'femnist.cnn': (0.01, 62, 450),  # lr, num_classes, test_batch_size
    'femnist.2nn': (0.03, 62, 450),  # lr, num_classes
    'shakespeare.lstm': (0.6, 80, 80, 256, 128),  # lr, seq_len, num_classes, num_hidden, test_batch_size
    'celeba.cnn': (0.1, 2),  # lr, num_classes
    'synthetic.log_reg': (0.0003, 50, 60),  # lr, num_classes, input_dim
    'cifar10.cnn': (0.01, 10, 64),  # lr, num_classes, test_batch_size
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
LOSS_KEY = 'loss'
SET_KEY = 'set'
CORRUPTED_KEY = "corrupted"