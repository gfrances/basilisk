#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR

from defaults import generate_experiment
from tarski.dl import PrimitiveRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept


def experiment(experiment_name=None):
    domain = "domain.pddl"
    domain_dir = "blocks"
    # benchmark_dir = BENCHMARK_DIR
    benchmark_dir = PYPERPLAN_BENCHMARK_DIR
    # domain_dir = "gripper"

    prob01 = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances="task01.pddl",
        num_states=500, num_sampled_states=None, random_seed=12,
        max_concept_size=20, max_concept_grammar_iterations=3,
        concept_generator=None, parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,)




    parameters = {
        "task01": prob01,
    }.get(experiment_name or "test")

    return generate_experiment(domain_dir, domain, **parameters)


def add_domain_parameters(language):
    return []


def feature_namer(feature):
    s = str(feature)
    return {
    }.get(s, s)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
