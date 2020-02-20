#!/usr/bin/env python

import argparse
import logging
import math
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
        for line in goal_file:
            nodes = list(map(int, line.split()))
            source = nodes[0]
            dist[source] = INFINITY
            for v in nodes[1:]:
                transitions[source].append(v)

    for g in goals:
        dist[g] = 0

    return dist

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

    logging.info('Running script "{}"'.format(domain_file))
    logging.info('Running configuration "{}"'.format(args.configuration))
    logging.warning(
        "Experiment configuration should NOT be an incremental experiment.")

    with open(args.output, 'w') as f:
        logging.info('Running experiment. Output saved to file "{}".'.format(args.output))
        subprocess.call([domain_file, args.configuration, '1', '2', '3', '4'],
                        stdout=f)
        logging.info('Experiment finished.')
    with open(args.output, 'r') as f:
        for line in f:
            if "Using experiment directory" in line:
                exp_dir = line.replace("Using experiment directory ", '').replace('\n', '')
                logging.info('Experiment directory: "{}"'.format(exp_dir))
                break

    if exp_dir is None:
        logging.error('No experiment directory found!')
        sys.exit()


    h_star = compute_h_star(exp_dir)
