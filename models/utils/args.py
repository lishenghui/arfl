import argparse
import os

DATASETS = ['sent140', 'femnist', 'shakespeare', 'cifar10']
SIM_TIMES = ['small', 'medium', 'large']
METHODS = ['fedavg', 'rfa', 'arfl', 'mkrum', 'cfl']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='femnist',
                        required=True)
    parser.add_argument('--method',
                        help='name of dataset;',
                        choices=METHODS,
                        type=str,
                        default='arfl')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='cnn',
                        required=True)
    parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch-size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=8)
    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=1)
    parser.add_argument('--metrics_name',
                        help='name for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('--metrics-dir',
                        help='dir for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('--use-val-set',
                        help='use validation set;',
                        action='store_true')
    parser.add_argument('--reg_weight', help='weight of regularization term', type=float, required=False)
    parser.add_argument('-pc', help='proportion of clients to be corrupted', type=float, default=0.0)
    parser.add_argument('-ps', help='proportion of sample in each client to be corrupted', type=float, default=1.)
    parser.add_argument('--setup_clients', help='proportion of setup clients', type=float, default=1.)
    parser.add_argument('--attack_type', help='Attacking option', type=int, default=3)
    parser.add_argument('--num_actors', help='total number of actors', type=int, default=4)
    parser.add_argument('--num_gpus', help='number of gpus', type=int, default=1)
    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                                        help='None for FedAvg, else fraction;',
                                        type=float,
                                        default=None)
    epoch_capability_group.add_argument('--num-epochs',
                                        help='number of epochs when clients train on data;',
                                        type=int,
                                        default=2)

    parser.add_argument('-t',
                        help='simulation time: small, medium, or large;',
                        type=str,
                        choices=SIM_TIMES,
                        default='large')
    parser.add_argument('--lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=0.01,
                        required=False)

    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--decay', dest='decay', action='store_true')
    flag_parser.add_argument('--no-decay', dest='decay', action='store_false')

    load_flag_parser = parser.add_mutually_exclusive_group(required=False)
    load_flag_parser.add_argument('--loadmodel', dest='loadmodel', action='store_true')
    load_flag_parser.add_argument('--no-loadmodel', dest='loadmodel', action='store_false')
    
    parser.set_defaults(decay=False)
    parser.set_defaults(loadmodel=False)
    return parser.parse_args()



ARGS = parse_args()

if ARGS.num_gpus == 0:
    GPU_PER_ACTOR = 0
    # CPU_PER_ACTOR = 1
else:
    GPU_PER_ACTOR = (ARGS.num_gpus - 0.05) / ARGS.num_actors
    # CPU_PER_ACTOR = (multiprocessing.cpu_count() - 0.05) / ARGS.num_actors

LOG_FILENAME = os.path.join(ARGS.metrics_dir, ARGS.metrics_name + "_state.csv")
