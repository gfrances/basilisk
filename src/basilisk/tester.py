#!/usr/bin/env python3
import logging
import sys

import sltp
from sltp.extensions import UniverseIndex, ExtensionCache
from sltp.features import parse_pddl, try_number
from sltp.features import Model as FeatureModel
from tarski.dl import UniversalConcept, EmptyConcept
from tarski.syntax.transform.errors import TransformationError
from tarski.syntax.transform.simplifications import transform_to_ground_atoms
from tarski import fstrips

from . import PYPERPLAN_DIR
from .search import create_pyperplan_hill_climbing_with_embedded_heuristic


def compute_goal_denotation(problem, use_goal_denotation):
    if not use_goal_denotation:
        return []

    try:
        return transform_to_ground_atoms(problem.goal)
    except TransformationError:
        logging.error("Cannot create goal concepts when problem goal is not a conjunction of ground atoms")
        raise


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

    # Parse the domain & instance with Tarski
    problem, language, generic_constants, types = parse_pddl(domain, instance)
    static_atoms = compute_static_atoms(problem)
    goal_denotation = compute_goal_denotation(problem, use_goal_denotation)
    processor = DenotationProcessor(language, goal_denotation)
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
        cache = processor.create_cache_for_state(translated_with_statics)
        model = FeatureModel(cache)

        h = 0
        for feature, weight in heuristic_parameters:
            denotation = int(model.compute_feature_value(feature, 0))  # State ID needs to be 0!
            delta = weight * denotation
            logging.info("\t\tFeature \"{}\": {} = {} * {}".format(feature, delta, weight, denotation))
            h += delta

        logging.info("\th(s)={} for state {}".format(h, state))
        return h

    return pyperplan_concept_based_heuristic


def translate_atom(atom):
    assert atom[0] == '(' and atom[-1] == ')'
    return atom[1:-1].split(' ')


def translate_state(state):
    """ Translate a pyperplan-like state into a list with the format required by SLTP's concept denotation processor """
    return [translate_atom(atom) for atom in state]


class DenotationProcessor(object):
    def __init__(self, language, goal_denotation):
        self.language = language
        self.universe_sort = language.get_sort('object')
        self.top = UniversalConcept(self.universe_sort.name)
        self.bot = EmptyConcept(self.universe_sort.name)
        self.universe = compute_universe_from_pddl_model(language)
        self.processor = sltp.SemanticProcessor(language, [], self.universe, self.top, self.bot)
        self.goal_denotation = goal_denotation

    def create_cache_for_state(self, state):
        cache = ExtensionCache(self.universe, self.top, self.bot)
        # We give an arbitrary state ID to 0 to the only state we're interested in
        self.processor.register_state_on_cache(0, state, cache, self.goal_denotation)
        return cache


def compute_universe_from_pddl_model(language):
    """ Compute a Universe Index from the PDDL model """
    universe = UniverseIndex()
    for obj in language.constants():
        universe.add(try_number(obj.symbol))
    universe.finish()  # No more objects possible
    return universe


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
