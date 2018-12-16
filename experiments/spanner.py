#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR

from sltp.util.misc import update_dict
from defaults import generate_experiment
from tarski.dl import PrimitiveRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept


def experiment(experiment_name=None):
    domain = "domain.pddl"
    domain_dir = "spanner"
    benchmark_dir = BENCHMARK_DIR

    spanner_1 = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        # instances=["prob-3-3-3-1540903410.pddl"],  # TODO HC on the training instance doesn't work
        # instances=["prob-3-3-3-1540903410.pddl", "prob-4-4-3-1540907456.pddl", "prob-4-3-3-1540907466.pddl"],  # TODO Doesn't work, fix this
        instances=["prob-4-3-3-1540907466.pddl"],
        test_instances=["prob-10-10-10-1540903568.pddl", "prob-15-10-8-1540913795.pddl"],
        test_domain=domain,
        # instances="task01.pddl",
        num_states=300,  # num_sampled_states=None,  random_seed=12,
        max_concept_size=10, max_concept_grammar_iterations=3,
        concept_generator=None,
        # concept_generator=generate_chosen_concepts,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    parameters = {
        "spanner_1": spanner_1,
    }.get(experiment_name or "test")

    return generate_experiment(domain_dir, domain, **parameters)


def generate_chosen_concepts(lang):
    """  """
    return [], [], []  # atoms, concepts, roles


def add_domain_parameters(language):
    return []


def feature_namer(feature):
    s = str(feature)
    return {
        "card[Exists(at,Not({roomb}))]": "nballs-A",
    }.get(s, s)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
