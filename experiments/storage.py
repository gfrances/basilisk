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
    domain_dir = "storage"
    benchmark_dir = DOWNWARD_BENCHMARKS_DIR

    storage_incremental = dict(
        experiment_class=IncrementalExperiment,
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances = ['p01.pddl',
                     'p02.pddl',
                     'p03.pddl',
                     'p04.pddl',
                     'p05.pddl',
                     'p06.pddl',
                     'p07.pddl',],
        test_instances=['p08.pddl',
                        'p09.pddl',
                        'p10.pddl',
                        'p11.pddl',
                        'p12.pddl',
                        'p13.pddl',
                        'p14.pddl',
                        'p15.pddl',
                        'p16.pddl',
                        'p17.pddl',
                        'p18.pddl',
                        'p19.pddl',
                        'p20.pddl',
                        'p21.pddl',
                        'p22.pddl',
                        'p23.pddl',
                        'p24.pddl',
                        'p25.pddl',
                        'p26.pddl',
                        'p27.pddl',
                        'p28.pddl',
                        'p29.pddl',
                        'p30.pddl',],
        test_domain=domain,
        distance_feature_max_complexity=6,
        num_states=10000,
        initial_sample_size=100,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=16, concept_bound_step=2,
        batch_refinement_size=50,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    parameters = {
        "storage_incremental": storage_incremental,
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
