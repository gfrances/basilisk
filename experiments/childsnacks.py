#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR, DOWNWARD_BENCHMARKS_DIR
from basilisk.incremental import IncrementalExperiment

from sltp.util.misc import update_dict
from defaults import generate_experiment
from tarski.dl import PrimitiveRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept


def experiment(experiment_name=None):
    domain = "domain.pddl"
    domain_dir = "childsnacks"
    benchmark_dir = BENCHMARK_DIR

    childsnacks_1_incremental = dict(
        experiment_class=IncrementalExperiment,
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances = ['child-snack_pfile01.pddl',],
        test_instances=['child-snack_pfile01-2.pddl'],
        test_domain=domain,
        #distance_feature_max_complexity=10,
        num_states=200,
        initial_sample_size=8,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=5,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    parameters = {
        "childsnacks_1_incremental": childsnacks_1_incremental,
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
