#!/usr/bin/python

import sys


def read_transition_file(path):
    transitions = []
    adj_list = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            state, targets = parts[0], parts[1:]
            adj_list['s' + state] = []
            for transition in targets:
                transitions.append(("s" + state, "s" + transition))
                adj_list['s' + state].append("s" + transition)
    return transitions, adj_list


def read_features_file(path):
    features_per_state = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            state = 's' + parts[0]
            features_per_state[state] = [int(feature) for feature in parts[1:]]
    assert features_per_state
    return features_per_state, \
        len(next(iter(features_per_state.values())))  # get the len of any element, they'll all be equal


def read_state_set(path):
    goal_states = []
    try:
        f = open(path, 'r')
    except IOError:
        print("Could not read file:", path)
        sys.exit()

    with f:
        for line in f.readlines():
            for state in line.split():
                goal_states.append('s' + state)
    return set(x for x in goal_states)


def read_complexity_file(path):
    feature_complexity = []
    names = []
    try:
        f = open(path, 'r')
    except IOError:
        print("Could not read file:", path)
        sys.exit()

    with f:
        for line in f.readlines():
            splitted = line.split()
            names.append(' '.join(splitted[:-1]))
            feature_complexity.append(int(splitted[-1]))
    return feature_complexity, names
