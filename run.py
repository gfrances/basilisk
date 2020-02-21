#!/usr/bin/env python

import argparse
import logging
import math
import numpy as np
import os
import shutil
import subprocess
import sys
import time

from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate training data for generalized potential '
                    'heuristics.')
    parser.add_argument('-d', '--domain', default=None, required=True,
                        help="Name of the domain script (without .py).")
    parser.add_argument('-c', '--configuration', default=None, required=True,
                        help="Name of the configuration.")
    parser.add_argument('-o', '--output', default='output.log',
                        help="Output of experiment run.")
    parser.add_argument('--debug', action='store_true', help='Set DEBUG flag.')
    parser.add_argument('--distances', default='distances.dat',
                        help='File where h-star distances are output.')
    parser.add_argument('--features', default='features.dat',
                        help='Feature values by state.')
    parser.add_argument('--keep-files', action='store_true',
                        help='Keep intermediate files (e.g., h-star table). '
                             'Experiment files are always store under "runs/".')

    args = parser.parse_args()
    return args


def compute_h_star(exp_dir):
    '''
    Extracts h-star for every state sampled from the files in exp_dir
    '''

    INFINITY = math.inf
    dist = defaultdict(lambda: INFINITY)
    transitions = defaultdict(list)

    with open(exp_dir + '/goal-states.dat') as goal_file:
        # Read goal states
        for line in goal_file:
            goals = list(map(int, line.split()))
    assert len(goals) > 0

    with open(exp_dir + '/transition-matrix.dat') as goal_file:
        # Read transition file
        for line in goal_file:
            nodes = list(map(int, line.split()))
            source = nodes[0]
            dist[source] = INFINITY
            for v in nodes[1:]:
                transitions[v].append(source)

    queue = []
    for g in goals:
        dist[g] = 0
        queue.append(g)

    while len(queue) != 0:
        node = queue.pop(0)
        d = dist[node]
        for t in transitions[node]:
            if dist[t] > d + 1:
                dist[t] = d + 1
                queue.append(t)

    return dist


def define_nn_input_and_output(features_file, h_star):
    """
    Organize table of features and h-star values into data structures for Keras
    """
    features = defaultdict(list)
    with open(features_file, 'r') as f:
        for line in f:
            values = line.split()
            features[int(values[0])] = [int(x) for x in values[1:]]

    num_features = len(features[0]) - 1
    num_states = len(features)

    # Loop over all states adding the features and the h_star value in the
    # same order, discarding the state ID of them
    clean_features = []
    clean_h_star = []
    for s, f in features.items():
        clean_features.append(f)
        clean_h_star.append(h_star[s])
    return np.array(clean_features), np.array(clean_h_star)


def call_experiment(arguments):
    """
    Run experiment from the arguments passed in the command line.
    """
    logging.info('Running script "{}"'.format(domain_file))
    logging.info('Running configuration "{}"'.format(arguments.configuration))
    logging.warning(
        "Experiment configuration should NOT be an incremental experiment.")
    with open(arguments.output, 'w') as f:
        logging.info('Running experiment. Output saved to file "{}".'.format(
            arguments.output))
        subprocess.call([domain_file, arguments.configuration,
                         '1', '2', '3', '4'],
                        stdout=f)
        logging.info('Experiment finished.')


def get_experiment_directory(arguments):
    """
    Obtain experiment directory where all data files are stored.
    """
    directory = None
    with open(arguments.output, 'r') as f:
        for line in f:
            if "Using experiment directory" in line:
                directory = line.replace("Using experiment directory ",
                                       '').replace('\n', '')
                logging.info('Experiment directory: "{}"'.format(directory))
                break
    return directory


def write_feature_and_h_star_files(values, arguments):
    logging.info('Writing distances to "%s"' % arguments.distances)
    with open(arguments.distances, 'w') as df:
        for node, value in values.items():
            df.write('{} {}\n'.format(node, value))
    logging.info('Writing features to "%s"' % arguments.features)
    shutil.copyfile(exp_dir + '/feature-matrix.dat', arguments.features)


if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname)-8s- %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)

    domain_file = os.getcwd() + '/experiments/' + args.domain + '.py'
    if not os.path.isfile(domain_file):
        logging.error(
            'Error: Script file "%s" does not exist.\n' % domain_file)
        sys.exit()

    call_experiment(args)
    exp_dir = get_experiment_directory(args)

    if exp_dir is None:
        logging.error('No experiment directory found!')
        sys.exit()

    h_star = compute_h_star(exp_dir)
    if args.keep_files:
        write_feature_and_h_star_files(h_star, args)

    # Set up data for keras
    input_features, output_values = define_nn_input_and_output(args.features,
                                                               h_star)
    logging.info("Setting up input features and output values for NN.")
