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
    domain_dir = "logistics"
    benchmark_dir = PYPERPLAN_BENCHMARK_DIR

    # Logistics is currently  not working
    logistics_1 = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=["task01.pddl"],
        test_instances=["task02.pddl",],
        test_domain=domain,
        distance_feature_max_complexity=0,
        num_states=20000,
        num_sampled_states=None, random_seed=12,
        max_concept_size=10, max_concept_grammar_iterations=3,
        concept_generator=None,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    logistics_1_incremental = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=["task01.pddl"],
        test_instances=["task02.pddl",],
        test_domain=domain,
        distance_feature_max_complexity=0,
        num_states=20000,
        initial_sample_size=8,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=5,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    parameters = {
        "logistics_1": logistics_1,
        "logistics_1_incremental": logistics_1_incremental,
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
