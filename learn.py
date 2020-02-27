#!/usr/bin/env python

import argparse
import logging
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys

from collections import defaultdict
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, Input, ReLU, ELU, LeakyReLU, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.utils import plot_model
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler

# Global variables for customized loss function
H_STAR = []
TRANSITIONS = defaultdict(list)


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
    parser.add_argument('--neurons-multiplier', default=1, type=int,
                        help='Multiplier for the number of neurons in the '
                             'layers of the NN. The number of neurons '
                             'in each of the hidden layers will be the'
                             'total number of features * the multiplier.')
    parser.add_argument('--iterations', default=1, type=int,
                        help='Number of iterations of the iterative learning'
                             'to reduce number of used features.')
    parser.add_argument('--class-weights', action='store_true',
                        help='Use weighted class on training.')
    args = parser.parse_args()
    if not os.path.isdir(args.path):
        logging.error(
            'Error: Directory "%s" does not exist.\n' % args.path)
        sys.exit()
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
    TRANSITIONS = transitions

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

    H_STAR = dist
    return dist


def read_training_data(path):
    INFINITY = 2147483647
    features_file = path + '/feature-matrix.io'
    table = np.loadtxt(features_file)
    features = table[:, np.all(table != INFINITY, axis=0)]
    dist = compute_h_star(path)
    h_star = []
    final_features = []
    for i, f in enumerate(features):
        if not math.isinf(dist[i]):
            h_star.append(dist[i])
            final_features.append(f)

    np.savetxt(path + '/training-examples.csv', np.array(final_features), delimiter=',', fmt='%d')
    np.savetxt(path + '/training-labels.csv', np.array(h_star, dtype=int), delimiter=',', fmt='%d')
    return np.array(final_features), np.array(h_star)


def train_nn(model, X, Y, args):
    """
    Compile and train neural network
    """
    weights = None
    if args.class_weights:
        weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
    logging.info('Compiling NN before training')
    adam = Adam(learning_rate=0.001)
    model.compile(loss='mse', metrics=["mae"], optimizer=adam)
    logging.info('Training the NN....')
    history = model.fit(X, Y, epochs=args.epochs, batch_size=args.batch,
                        class_weight=weights,
                        #callbacks=[EarlyStopping(monitor='loss', patience=20)],
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

    # L2 was not used because it does not seem to be needed.
    # L1 alone performed better in the cases tested.
    for i in range(args.hidden_layers):
        tmp = hidden
        hidden = Concatenate()([hidden, last_hidden])
        hidden = Dense(nf * args.neurons_multiplier,
                       kernel_regularizer=l1(0.01))(hidden)
        #hidden = LeakyReLU()(hidden)
        last_hidden = tmp
    hidden = Dense(1, kernel_regularizer=l1(0.01))(hidden)
    hidden = Activation('linear')(hidden)
    return Model(inputs=input_layer, outputs=hidden)


def plot(history, X, Y, args):
    """
    Produce plots related to the learned heuristic function
    """
    epochs = args.epochs

    # Plot error curve
    mse_loss = history.history['loss']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 10])
    ax.set_xlim([0, epochs])
    ax.plot(mse_loss)
    plt.show()

    # Scatter plot comparing h*(X-axis) to the predicted values (Y-axis)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    yp = history.model.predict(X)
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
    plt.show()

    # Plot histogram comparing h* values (blue) to the predicated values (
    # orange)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axvline(np.mean(Y))
    min_value = int(min(np.min(Y), np.floor(np.min(yp))))
    max_value = int(max(np.max(Y), np.ceil(np.max(yp))))
    bin_number = max_value - min_value
    bins = np.linspace(min_value, max_value, bin_number)
    ax.hist(Y, bins, alpha=0.5, label='h*')
    ax.hist(yp, bins, alpha=0.5, label='predicted')
    fig.legend()
    plt.show()

    for index, pair in enumerate(zip(Y, yp)):
        if pair[1] < 0:
            print(index, pair)

    # Plot weights
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.model.layers[-2].get_weights()[0])
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Weight")
    ax.set_title(args.batch)
    plt.show()

    return


def compute_inconsistent_states(i, o):
    logging.info('Checking for inconsistent input...')
    for s1, h1 in zip(i, o):
        for s2, h2 in zip(i, o):
            if h1 != h2 and np.array_equal(s1, s2):
                logging.warning("At least one pair of states with different "
                                "h* value but same feature denotations!")
                return


def iterative_learn(input_features, output_values):
    # Set up data for keras
    num_training_examples, num_features = input_features.shape
    logging.info(
        'Total number of training examples: %d' % num_training_examples)
    logging.info('Total number of input features: %d' % num_features)

    # compute_inconsistent_states(input_features, output_values)

    logging.info('Creating the NN model')
    model = create_nn(args, num_features)
    print(model.summary())

    history = train_nn(model, input_features, output_values, args)
    if args.plot:
        plot(history, input_features, output_values, args)
    return history


def get_significant_weights(history):
    lst = []
    for index, e in enumerate(history.model.layers[-2].get_weights()[0]):
        if abs(e) > 0.1:
            lst.append(index)
    return lst


def filter_input(indices, input_features):
    selection = [False for x in range(input_features.shape[1])]
    for feat in indices:
        selection[feat] = True
    return input_features[:, selection]


if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname)-8s- %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)

    logging.info('Reading training data from %s' % args.path)
    input_features, output_values = read_training_data(args.path)

    # input_features = np.delete(input_features, (39,135,647), axis=0)
    # output_values = np.delete(output_values, (39, 135, 647), axis=0)
    # #input_features = np.delete(input_features, (135), axis=1)
    # #input_features = np.delete(input_features, (647), axis=1)

    for i in range(args.iterations):
        history = iterative_learn(input_features, output_values)
        indices = get_significant_weights(history)
        logging.info('Useful features: {}'.format(indices))
        input_features = filter_input(indices, input_features)
