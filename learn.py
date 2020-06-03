#!/usr/bin/env python

import argparse
import logging
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import random
import sys

from collections import defaultdict

from basilisk.tester import import_and_run_pyperplan
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, Input, ReLU, ELU, LeakyReLU, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.utils import plot_model
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from tarski.dl import ConceptCardinalityFeature, EmpiricalBinaryConcept

from sltp.language import parse_pddl

# Global variables
from sltp.util.serialization import unserialize_feature

INFINITY = 2147483647  # C++ infinity value

H_STAR = []
TRANSITIONS = defaultdict(list)

VERBOSE_LEVEL = 1


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Learn a heuristic based on a training data')
    parser.add_argument('-p', '--training-data', default=None,
                        required=True,
                        help="Path of the directory containing the training "
                             "data. This directory should contain the files "
                             "'features.dat' and 'distances.dat'")
    parser.add_argument('-t', '--test-data', default=None,
                        required=True,
                        help="Path of the directory containing the test "
                             "data. This directory should contain the domain "
                             "and instances files.")
    parser.add_argument('--split-training-data', nargs='*', type=int,
                        default=None,
                        help="SHIFT TRAINING_PARTS VALIDATION_PARTS. "
                             "Splits the loaded training data "
                             "into training, validation data. The nn is "
                             "trained on the training "
                             "data. After each epoch the performance of the "
                             "nn is evaluated on the validation data. "
                             "Depending on this evaluation, training "
                             "parameters can be tuned. "
                             "*_PARTS defines how many parts of the original "
                             "training data are given to this split. SHIFT"
                             "deterministically defines which parts of the "
                             "data are given to which new data set. SHIFT is"
                             "a value between 0 and sum(*_PARTS).")
    parser.add_argument('--debug', action='store_true', help='Set DEBUG flag.')
    parser.add_argument('--plot', action='store_true',
                        help='Plot learning histograms and curves.')
    parser.add_argument('--epochs', default=5000, type=int,
                        help='Number of the epochs for the NN training.')
    parser.add_argument('--batch', default=250, type=int,
                        help='Batch training size.')
    parser.add_argument('--hidden-layers', default=0, type=int,
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
    parser.add_argument('--keras-verbose', default=1, type=int,
                        help='Keras verbose level (0, 1, or 2).')
    parser.add_argument('--seed', type=int, default=42,
                        help="set a random seed for python (not for keras)")
    args = parser.parse_args()

    random.seed(args.seed)

    check_path_exists(args.training_data)
    check_path_exists(args.test_data)

    if args.split_training_data is not None:
        check_argument_value(
            len(args.split_training_data) == 3,
            "Invalid number of arguments for --split-training-data")
        check_argument_value(
            all(x >= 0 for x in args.split_training_data),
            "All arguments for --split-training-data has to be positive")
        check_argument_value(
            args.split_training_data[0] < sum(
                x for x in args.split_training_data[1:]),
            "SHIFT of --split-training-data has to be less than sum(*_PARTS)")

    check_argument_value(
        args.hidden_layers == 0,
        "The feature selection does not work with hidden layers")
    return args


def check_argument_value(condition, message):
    if not condition:
        logging.error(message)
        sys.exit()


def check_path_exists(path):
    if not os.path.isdir(path):
        logging.error(
            'Error: Directory "%s" does not exist.\n' % path)
        sys.exit()


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
    """
    Read the input data and returns a matrix with the feature denotation
    in every state, the h-star value, and a list of strings representing
    the features.
    """

    # First read feature matrix and align it with H*.
    feature_matrix_file = path + '/feature-matrix.io'
    table = np.loadtxt(feature_matrix_file)
    features = table[:, np.all(table != INFINITY, axis=0)]
    dist = compute_h_star(path)
    h_star = []
    finite_features_denotations = []
    for i, f in enumerate(features):
        if not math.isinf(dist[i]) and dist[i] != 0:
            h_star.append(dist[i])
            finite_features_denotations.append(f)

    # Read the features info to extract the strings representing them
    feature_info_file = path + '/feature-info.io'
    with open(feature_info_file) as f:
        feature_line = f.readline()
        feature_strings = feature_line.split('\t')

    assert (feature_strings is not None)

    np.savetxt(path + '/training-examples.csv', np.array(finite_features_denotations), delimiter=',', fmt='%d')
    np.savetxt(path + '/training-labels.csv', np.array(h_star, dtype=int), delimiter=',', fmt='%d')
    return np.array(finite_features_denotations), np.array(h_star), np.array(feature_strings)


def split_training_data(x, y, args):
    assert x.shape[0] == y.shape[0]
    if args.split_training_data is None:
        return (x, y), None
    shift = args.split_training_data[0]
    parts = args.split_training_data[1:]
    nb_parts = sum(x for x in parts)
    assert (0 <= shift < nb_parts)
    assert (all(x >= 0 for x in parts))

    # Define for each data set which folds (aka parts) of the data belong to it
    start = 0
    assigned_folds = []
    for part in parts:
        assigned_folds.append([(i + start + shift) % nb_parts
                               for i in range(part)])
        start += part

    # Distribute the samples (via their index) to each data set
    indices = np.arange(y.shape[0])
    random.shuffle(indices)
    part_length = int(y.shape[0] / nb_parts)

    def _get_start_end(_fold):
        assert _fold < nb_parts
        if _fold == nb_parts - 1:
            return _fold * part_length, y.shape[0]
        else:
            return _fold * part_length, (_fold + 1) * part_length

    distributed = []
    for af in assigned_folds:
        distributed.append([])
        for fold in af:
            start, end = _get_start_end(fold)
            distributed[-1].append(indices[start:end])
    distributed = [np.concatenate(d) for d in distributed]

    # Distribute samples with determined indices
    return [(x[d], y[d]) for d in distributed]


def train_nn(model, data_train, data_valid, args):
    """
    Compile and train neural network
    """
    X, Y = data_train
    weights = None
    if args.class_weights:
        weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
    logging.info('Compiling NN before training')
    adam = Adam(learning_rate=0.001)
    model.compile(loss='mse', metrics=["mae"], optimizer=adam)
    logging.info('Training the NN....')
    history = model.fit(X, Y, epochs=args.epochs, batch_size=args.batch,
                        class_weight=weights,
                        verbose=VERBOSE_LEVEL,
                        validation_data=data_valid,
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


def plot(history, data, args):
    """
    Produce plots related to the learned heuristic function
    """
    X, Y = data
    epochs = args.epochs

    # Plot error curve
    mse_loss = history.history['loss']
    mse_val_loss = history.history.get('val_loss')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 10])
    ax.set_xlim([0, epochs])
    ax.plot(mse_loss, label="loss")
    if mse_val_loss is not None:
        ax.plot(mse_val_loss, label="validation loss")
    ax.legend()
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


def iterative_learn(args, data_train, data_valid):
    # Set up data for keras
    num_training_examples, num_features = data_train[0].shape
    num_validation_examples = (0 if data_valid is None else len(data_valid[0]))
    logging.info(
        'Total number of training examples: %d' % num_training_examples)
    logging.info(
        'Total number of validation examples: %d' % num_validation_examples)
    logging.info('Total number of input features: %d' % num_features)

    # compute_inconsistent_states(input_features, output_values)

    logging.info('Creating the NN model')
    model = create_nn(args, num_features)
    print(model.summary())

    history = train_nn(model, data_train, data_valid, args)
    if args.plot:
        plot(history, data_train, args)
    return history


def get_significant_weights(history):
    significant_weights = []
    lst = []
    for index, w in enumerate(history.model.layers[-2].get_weights()[0]):
        if abs(w) > 0.1:
            significant_weights.append(w)
            lst.append(index)
    return significant_weights, lst


def filter_input(indices, features_names, *data_sets):
    assert len(data_sets) > 0
    assert data_sets[0] is not None

    selection = [False for _ in range(data_sets[0][0].shape[1])]
    new_feature_names = []
    for feat in indices:
        selection[feat] = True
        new_feature_names.append(features_names[feat])

    return [None if ds is None else (ds[0][:, selection], ds[1])
            for ds in data_sets], new_feature_names


def read_test_instances_list(path):
    domain = None
    instances = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for f in filenames:
            if not f.endswith('.pddl'):
                continue
            if 'domain' in f:
                domain = os.path.join(dirpath, f)
            else:
                instances.append(os.path.join(dirpath, f))

    return domain, instances


def create_language(domain):
    _, language, _ = parse_pddl(domain)
    return language


def add_gripper_domain_parameters(language):
    # Hack for now
    return [language.constant("roomb", "object")]


def main():
    args = parse_arguments()
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname)-8s- %(message)s",
                        level=logging.DEBUG if args.debug else logging.INFO)

    logging.info('Reading training data from %s' % args.training_data)
    input_features, output_values, features_names = \
        read_training_data(args.training_data)
    data_train, data_valid = \
        split_training_data(input_features, output_values, args)

    test_domain, test_instances = read_test_instances_list(args.test_data)

    VERBOSE_LEVEL = args.keras_verbose

    weights = None
    for i in range(args.iterations):
        history = iterative_learn(args, data_train, data_valid)
        weights, indices = get_significant_weights(history)
        if len(indices) == 0:
            logging.error("No useful features remaining.")
            sys.exit()

        logging.info('Useful features: {}'.format(indices))
        (data_train, data_valid), features_names = filter_input(
            indices, features_names, data_train, data_valid)

    assert (weights is not None and len(weights) == len(features_names))
    print("Weighted features found:")
    for f, w in zip(features_names, weights):
        print("{} * {}".format(np.asscalar(w), f))

    language = create_language(test_domain)

    heuristic = create_potential_heuristic_from_parameters(features_names,
                                                           weights,
                                                           language)

    test_instances.sort()
    for i in test_instances:
        logging.info("Solving {}".format(i))
        import_and_run_pyperplan(test_domain, i, heuristic,
                                 None, gbfs=True)

    logging.debug('Exiting script.')


class ConceptBasedPotentialHeuristic:
    def __init__(self, parameters):
        def transform(feature):
            if isinstance(feature, EmpiricalBinaryConcept):
                return ConceptCardinalityFeature(feature.c)
            return feature

        self.parameters = [(transform(feature), weight) for feature, weight in parameters]

    def value(self, model):
        h = 0
        for feature, weight in self.parameters:
            denotation = int(model.denotation(feature))
            delta = weight * denotation
            # logging.info("\t\tFeature \"{}\": {} = {} * {}".format(feature, delta, weight, denotation))
            h += delta

        # logging.info("\th(s)={} for state {}".format(h, state))
        return h

    def __str__(self):
        return " + ".join("{}·{}".format(weight, feature) for feature, weight in self.parameters)


def create_potential_heuristic_from_parameters(features, weights, language):
    selected_features = [unserialize_feature(language, str(f).rstrip('\n')) for f in features]
    selected_weights = [np.asscalar(w) for w in weights]
    return ConceptBasedPotentialHeuristic(list(zip(selected_features, selected_weights)))


if __name__ == '__main__':
    main()