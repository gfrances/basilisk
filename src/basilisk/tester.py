#!/usr/bin/env python3
import logging
import sys

from . import PYPERPLAN_DIR
from .search import create_pyperplan_hill_climbing_with_embedded_heuristic


def run_pyperplan(pyperplan, domain, instance, pyerplan_heuristic):
    # Let's fake a pyperplan command-line arguments object so that
    # we don't have to worry about default parameters. We use "gbf" but will override that next.
    args = pyperplan.parse_args([domain, instance])

    # And now we inject our desired search and heuristic functions
    args.forced_search = create_pyperplan_hill_climbing_with_embedded_heuristic(pyerplan_heuristic)

    # And run pyperplan!
    pyperplan.main(args)


def import_and_run_pyperplan(domain, instance, pyerplan_heuristic):
    sys.path.insert(0, PYPERPLAN_DIR)
    import pyperplan
    run_pyperplan(pyperplan, domain, instance, pyerplan_heuristic)
    sys.path = sys.path[1:]


def compute_feature_denotation(feature, state):
    assert False
    # TODO - Implement the computation of the heuristic here.


def create_heuristic(heuristic_parameters):
    """ Create a pyperplan-like concept-based heuristic with the given features and weights"""
    def pyperplan_concept_based_heuristic(state):
        h = 0
        for feature, weight in heuristic_parameters:
            h += weight * compute_feature_denotation(feature, state)
        return h

    return pyperplan_concept_based_heuristic


def run(config, data, rng):
    learned_heuristic = data.learned_heuristic
    selected_features = [data.features[f_idx] for f_idx, _ in learned_heuristic]
    selected_weights = [f_weight for _, f_weight in learned_heuristic]
    heuristic_parameters = zip(selected_features, selected_weights)

    pyerplan_heuristic = create_heuristic(heuristic_parameters)
    for instance in config.test_instances:
        import_and_run_pyperplan(config.test_domain, instance, pyerplan_heuristic)

    # Return those values that we want to be persisted between different steps
    return dict(
    )
