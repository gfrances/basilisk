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
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate training data for generalized potential '
                    'heuristics.')
    parser.add_argument('-d', '--domain', default=None, required=True,
                        help="Name of the domain script (without .py).")
    parser.add_argument('-c', '--configuration', default=None, required=True,
                        help="Name of the configuration.")
    parser.add_argument('-o', '--output', default='output.log',
                        help='File where the experiment stdout is saved.')
    parser.add_argument('-p', '--path', default=None, required=True,
                        help="Path to the directory where the training "
                             "data is stored.")
    parser.add_argument('--debug', action='store_true', help='Set DEBUG flag.')

    args = parser.parse_args()
    return args


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


def write_feature_and_h_star_files(path, values, exp_dir):
    logging.info('Writing training data to "%s"' % path)
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info('Writing h-star values to "%s"' % (path + '/distances.dat'))
    with open(path + '/distances.dat', 'w') as df:
        for node, value in values.items():
            df.write('{} {}\n'.format(node, value))
    logging.info('Writing feature matrix to "%s"' % (path + '/features.dat'))
    shutil.copyfile(exp_dir + '/feature-matrix.dat', path+'/features.dat')

    # Copying more useful files to the training data
    logging.info('Writing features info to "%s"' % (path + '/features-info.dat'))
    shutil.copyfile(exp_dir + '/feature-info.dat', path + '/features-info.dat')

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
    logging.info('Experiment finished.')
    exp_dir = get_experiment_directory(args)

    if exp_dir is None:
        logging.error('No experiment directory found!')
        sys.exit()

    h_star = compute_h_star(exp_dir)
    write_feature_and_h_star_files(args.path, h_star, exp_dir)