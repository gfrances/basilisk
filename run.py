#!/usr/bin/env python

import argparse
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
import sys
import time

from collections import defaultdict
from keras.models import Model, Sequential
from keras.layers import Dense, Input, ReLU, ELU, LeakyReLU, Concatenate
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Learn a heuristic based on a training data')
    parser.add_argument('-p', '--path', default=None,
                        required=True,
                        help="Path of the directory containing the training "
                             "data. This directory should contain the files "
                             "'features.dat' and 'distances.dat'")
    parser.add_argument('--debug', action='store_true', help='Set DEBUG flag.')
    parser.add_argument('--plot', action='store_true',
                        help='Plot learning histograms and curves.')
    parser.add_argument('--epochs', default=5000, type=int,
                        help='Number of the epochs for the NN training.')
    parser.add_argument('--batch', default=250, type=int,
                        help='Batch training size.')
    parser.add_argument('--hidden-layers', default=2, type=int,
                        help='Number of hidden layers.')

    args = parser.parse_args()
    if not os.path.isdir(args.path):
        logging.error(
            'Error: Directory "%s" does not exist.\n' % args.path)
        sys.exit()
    return args


def read_training_data(path):
    features_file = path + '/features.dat'
    features = defaultdict(list)
    with open(features_file, 'r') as f:
        logging.info('Reading features from "%s"' % features_file)
        for line in f:
            values = line.split()
            features[int(values[0])] = [int(x) for x in values[1:]]

    distances_file = path + '/distances.dat'
    h_star = dict()
    with open(distances_file, 'r') as f:
        logging.info('Reading h-star distances from "%s"' % distances_file)
        for line in f:
            values = line.split()
            assert len(values) == 2
            if values[1] != 'inf':
                h_star[int(values[0])] = int(values[1])
            else:
                # Remove dead-ends from training data
                features.pop(int(values[0]), None)

    return features, h_star


def define_nn_input_and_output(features, h_star):
    """
    Organize table of features and h-star values into data structures for Keras
    """

    # Loop over all states adding the features and the h_star value in the
    # same order, discarding the state ID of them
    clean_features = []
    clean_h_star = []
    for s, f in features.items():
        clean_features.append(f)
        clean_h_star.append(h_star[s])
    return np.array(clean_features), np.array(clean_h_star)


def train_nn(model, X, Y, epochs):
    """
    Compile and train neural network
    """
    uv, uc = np.unique(Y, return_counts=True)
    weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y), Y)
    logging.info('Compiling NN before training')
    model.compile(loss='mse', metrics=["mae"], optimizer='adam')
    logging.info('Training the NN....')
    history = model.fit(X, Y, epochs=epochs, batch_size=args.batch,
                        callbacks=[EarlyStopping(monitor='loss', patience=50)],
                        class_weight = weights
                        )
    logging.info('Finished NN training.')

    return history


def create_nn(args, nf):
    """
    Create neural network architecture
    """
    input_layer = Input(shape=(nf,))
    hidden = input_layer
    last_hidden = input_layer
    for i in range(args.hidden_layers + 1):
        tmp = hidden
        hidden = Concatenate()([hidden, last_hidden])
        hidden = Dense(nf, kernel_regularizer="l1_l2")(hidden)
        hidden = LeakyReLU()(hidden)
        last_hidden = tmp
    hidden = Dense(1, kernel_regularizer="l1_l2")(hidden)
    hidden = ELU()(hidden)
    return Model(inputs=input_layer, outputs=hidden)


def plot(history, X, Y, epochs):
    """
    Produce plots related to the learned heuristic function
    """
    # Plot error curve
    mse_loss = history.history['loss']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 5])
    ax.set_xlim([0, epochs])
    ax.plot(mse_loss)
    fig.show()

    # Scatter plot comparing h*(X-axis) to the predicted values (Y-axis)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    yp = model.predict(X)
    ax.scatter(Y, yp)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('h*')
    ax.set_ylabel('Predicted h')
    fig.show()


if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname)-8s- %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)

    logging.info('Reading training data from %s' % args.path)
    features, h_star = read_training_data(args.path)

    # Set up data for keras
    logging.info("Setting up input features and output values for NN.")
    input_features, output_values = define_nn_input_and_output(features, h_star)
    num_training_examples, num_features = input_features.shape
    logging.info(
        'Total number of training examples: %d' % num_training_examples)
    logging.info('Total number of input features: %d' % num_features)

    logging.info('Creating the NN model')
    model = create_nn(args, num_features)

    history = train_nn(model, input_features, output_values, args.epochs)
    if args.plot:
        plot(history, input_features, output_values, args.epochs)
