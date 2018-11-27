#!/usr/bin/env python3
import sys

from sltp.features import create_denotation_processor
from tarski import fstrips

from . import PYPERPLAN_DIR
from .search import create_pyperplan_hill_climbing_with_embedded_heuristic


def state_as_atoms(state):
    """ Transform any state (i.e. interpretation) into a flat set of tuples,
    one per ground atom that is true in the state """
    atoms = set()
    for signature, elems in state.predicate_extensions.items():
        name = signature[0]
        for elem in elems:
            atoms.add((name, ) + tuple(o.symbol for o in elem))

    return atoms


def compute_static_atoms(problem):
    init_atoms = state_as_atoms(problem.init)
    index = fstrips.TaskIndex(problem.language.name, problem.name)
    # Not sure this code is too reliable... static detection seems to be buggy. Let's do that ourselves from fluent
    # detection then.
    index.process_symbols(problem)
    fluent_symbols = {x.head.symbol for x in index.fluent_symbols}
    static_atoms = {a for a in init_atoms if a[0] not in fluent_symbols}
    return static_atoms


def run_pyperplan(pyperplan, domain, instance, heuristic_parameters, use_goal_denotation):
    # Let's fake a pyperplan command-line arguments object so that
    # we don't have to worry about default parameters. We use "gbf" but will override that next.
    args = pyperplan.parse_args([domain, instance])

    # Parse the domain & instance and create a model generator
    problem, processor = create_denotation_processor(domain, instance, use_goal_denotation)

    static_atoms = compute_static_atoms(problem)
    pyerplan_heuristic = create_heuristic(processor, static_atoms, heuristic_parameters)

    # And now we inject our desired search and heuristic functions
    args.forced_search = create_pyperplan_hill_climbing_with_embedded_heuristic(pyerplan_heuristic)

    # And run pyperplan!
    pyperplan.main(args)


def import_and_run_pyperplan(domain, instance, heuristic_parameters, use_goal_denotation):
    sys.path.insert(0, PYPERPLAN_DIR)
    import pyperplan
    run_pyperplan(pyperplan, domain, instance, heuristic_parameters, use_goal_denotation)
    sys.path = sys.path[1:]


def create_heuristic(processor, static_atoms, heuristic_parameters):
    """ Create a pyperplan-like concept-based heuristic with the given features and weights"""
    def pyperplan_concept_based_heuristic(state):

        translated = translate_state(state)
        translated_with_statics = translated + list(static_atoms)
        model = processor.compute_model_from_state(translated_with_statics)

        h = 0
        for feature, weight in heuristic_parameters:
            denotation = int(model.compute_feature_value(feature, 0))  # State ID needs to be 0!
            delta = weight * denotation
            # logging.info("\t\tFeature \"{}\": {} = {} * {}".format(feature, delta, weight, denotation))
            h += delta

        # logging.info("\th(s)={} for state {}".format(h, state))
        return h

    return pyperplan_concept_based_heuristic


def translate_atom(atom):
    assert atom[0] == '(' and atom[-1] == ')'
    return atom[1:-1].split(' ')


def translate_state(state):
    """ Translate a pyperplan-like state into a list with the format required by SLTP's concept denotation processor """
    return [translate_atom(atom) for atom in state]


def run(config, data, rng):
    learned_heuristic = data.learned_heuristic
    selected_features = [data.features[f_idx] for f_idx, _ in learned_heuristic]
    selected_weights = [f_weight for _, f_weight in learned_heuristic]
    heuristic_parameters = list(zip(selected_features, selected_weights))

    use_goal_denotation = config.parameter_generator is None

    for instance in config.test_instances:
        import_and_run_pyperplan(config.test_domain, instance, heuristic_parameters, use_goal_denotation)

    # Return those values that we want to be persisted between different steps
    return dict()
