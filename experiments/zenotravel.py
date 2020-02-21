#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR
from basilisk.incremental import IncrementalExperiment

from defaults import generate_experiment
from tarski.dl import PrimitiveRole, GoalRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept, EqualConcept, StarRole, NullaryAtom, Concept, PrimitiveConcept


def experiment(experiment_name=None):
    domain = "domain.pddl"
    domain_dir = "zenotravel"
    benchmark_dir = BENCHMARK_DIR
    # benchmark_dir = PYPERPLAN_BENCHMARK_DIR
    # domain_dir = "gripper"

    experiments = dict()

    experiments['learn'] = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=['p01.pddl',
                   'p02.pddl',
                   'p03.pddl',
        ],
        test_instances=['p04.pddl',
                        'p05.pddl',
                        'p06.pddl',
                        'p07.pddl',
                        'p08.pddl',
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
                        'p20.pddl'],
        test_domain=domain,
        distance_feature_max_complexity=4,
        num_states=20000, num_sampled_states=None, random_seed=12,
        max_concept_size=4, max_concept_grammar_iterations=5,
        concept_generator=None, parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,)



    parameters = experiments[experiment_name]
    parameters["domain_dir"] = parameters.get("domain_dir", domain_dir)
    parameters["domain"] = parameters.get("domain", domain)
    return generate_experiment(**parameters)


def add_domain_parameters(language):
    return []


def feature_namer(feature):
    s = str(feature)
    return {
    }.get(s, s)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
