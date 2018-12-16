#!/usr/bin/env python3
import logging
import sys

from models import FeatureModel
from sltp.features import create_model_factory
from sltp.returncodes import ExitCode
from tarski import fstrips

from . import PYPERPLAN_DIR
from .search import create_pyperplan_hill_climbing_with_embedded_heuristic


def state_as_atoms(state):
    """ Transform any state (i.e. interpretation) into a flat set of tuples,
    one per ground atom that is true in the state """
    atoms = set()
    for signature, elems in state.list_all_extensions().items():
        name = signature[0]
        # We unwrap the tuples of Constants into tuples with their (string) names
        atoms.update((name, ) + tuple(o.symbol for o in elem) for elem in elems)
    return atoms


def compute_static_atoms(problem):
    init_atoms = state_as_atoms(problem.init)
    index = fstrips.TaskIndex(problem.language.name, problem.name)
    # Not sure the fstrips.TaskIndex code is too reliable... static detection seems to be buggy.
    # Let's better do that ourselves with a basic fluent detection routine.
    index.process_symbols(problem)
    fluent_symbols = {x.head.symbol for x in index.fluent_symbols}
    static_atoms = {a for a in init_atoms if a[0] not in fluent_symbols}
    return static_atoms


def run_pyperplan(pyperplan, domain, instance, heuristic, parameter_generator):
    # Let's fake a pyperplan command-line arguments object so that
    # we don't have to worry about default parameters. We use "gbf" but will override that next.
    args = pyperplan.parse_args([domain, instance])

    # Parse the domain & instance and create a model generator
    problem, model_factory = create_model_factory(domain, instance, parameter_generator)

    static_atoms = compute_static_atoms(problem)

    pyerplan_heuristic = create_pyperplan_heuristic(model_factory, static_atoms, heuristic)

    # And now we inject our desired search and heuristic functions
    args.forced_search = create_pyperplan_hill_climbing_with_embedded_heuristic(pyerplan_heuristic)

    # And run pyperplan!
    pyperplan.main(args)


def import_and_run_pyperplan(domain, instance, heuristic, parameter_generator):
    sys.path.insert(0, PYPERPLAN_DIR)
    import pyperplan
    run_pyperplan(pyperplan, domain, instance, heuristic, parameter_generator)
    sys.path = sys.path[1:]


def create_pyperplan_heuristic(model_factory, static_atoms, heuristic):
    """ Create a pyperplan-like concept-based heuristic with the given features and weights"""
    def pyperplan_concept_based_heuristic(state):

        translated = translate_state(state)
        translated_with_statics = translated + list(static_atoms)
        model = FeatureModel(model_factory.create_model(translated_with_statics))

        return heuristic.value(model)

    return pyperplan_concept_based_heuristic


def translate_atom(atom):
    assert atom[0] == '(' and atom[-1] == ')'
    return atom[1:-1].split(' ')


def translate_state(state):
    """ Translate a pyperplan-like state into a list with the format required by SLTP's concept denotation processor """
    return [translate_atom(atom) for atom in state]


def run(config, data, rng):
    for instance in config.test_instances:
        logging.info("Testing learnt heuristic on instance \"{}\"".format(instance))
        import_and_run_pyperplan(config.test_domain, instance, data.learned_heuristic, config.parameter_generator)

    # Return those values that we want to be persisted between different steps
    return ExitCode.Success, dict()
