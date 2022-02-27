"""Script to run the baselines."""
import datetime
import logging
import numpy as np
import os
import random
import ray
import tensorflow as tf
import time

import metrics.writer as metrics_writer
from baseline_constants import MAIN_PARAMS
from client_manager import ClientManager
from server import Server
from utils.args import ARGS


def main():
    start_time = time.time()
    args = ARGS
    ray.init(include_dashboard=False, num_gpus=args.num_gpus)
    log_filename = os.path.join(args.metrics_dir, args.metrics_name + '.csv')
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    
    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]
    args.clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]
    
    manager = ClientManager(args)
    clients = manager.setup_clients(args.setup_clients)
    clients.sort(key=lambda x: x.num_train_samples)
    
    manager.corrupt_clients()
    
    print('Clients in Total: %d' % len(clients))
    
    # Create server
    server = Server(clients, manager, args)
    if args.loadmodel:
        model = tf.keras.models.load_model(log_filename + "_model")
        server.set_model(model)
    client_ids, client_groups, num_train_samples, num_test_samples = manager.get_clients_info()
    total_train_samples = np.sum(list(num_train_samples.values()))
    for c, n in zip(clients, num_train_samples.values()):
        c.set_weight(float(n) / total_train_samples)
    
    # Initial status
    print('--- Random Initialization ---')
    server.test_model(0, set_to_use='train', log=False)
    # Simulate training
    for i in range(num_rounds):
        # Select clients to train this round
        server.select_clients(i, num_clients=clients_per_round)
        
        server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, round=i)
        aggregation_start = time.time()
        server.aggregate(args.method)
        
        if args.method == "arfl":
            server.update_alpha(i)
            
        print(datetime.datetime.now(),
              '--- Round %d of %d: Training %d Clients. Time cost in total %s. Aggregation time %s --- ' %
              (i + 1, num_rounds, clients_per_round, time.time() - start_time, time.time() - aggregation_start))
        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            test_stat_metrics = server.test_model(i, set_to_use='train')  # Evaluate training loss
            print_metrics(test_stat_metrics, num_train_samples, prefix='{}_'.format('train'))
            test_stat_metrics = server.test_model(i, set_to_use='test')
            print_metrics(test_stat_metrics, num_test_samples, prefix='{}_'.format('test'))
    
    # Save model when training ends
    server.save_model(log_filename + "_model")


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
