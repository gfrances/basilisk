#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR
from basilisk.incremental import IncrementalExperiment

from defaults import generate_experiment
from tarski.dl import PrimitiveRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept


def experiment(experiment_name=None):
    domain = "domain.pddl"
    domain_dir = "visitall"
    benchmark_dir = BENCHMARK_DIR
    # domain_dir = "gripper"

    problem02full = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=["problem02-full.pddl"], #, "problem03-full.pddl"],
        test_instances=["problem04-full.pddl"],
        test_domain=domain,
        distance_feature_max_complexity=10,
        num_states=500, num_sampled_states=None, random_seed=12,
        max_concept_size=8, max_concept_grammar_iterations=3,
        concept_generator=None, parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,)

    problem03full = dict(
        lp_max_weight=2,
        benchmark_dir=benchmark_dir,
        instances=["problem02-full.pddl",
                   "problem03-full.pddl"],
        test_instances=["problem04-full.pddl",
                        "problem05-full.pddl",
                        "problem06-full.pddl",
                        "problem07-full.pddl",
                        "problem08-full.pddl",
                        "problem09-full.pddl",
                        "problem10-full.pddl",
                        "problem11-full.pddl",],
        test_domain=domain,
        distance_feature_max_complexity=10,
        num_states=1000, num_sampled_states=None, random_seed=12,
        max_concept_size=8, max_concept_grammar_iterations=3,
        concept_generator=None, parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,)

    # Incremental version
    problem03full_incremental = dict(
        benchmark_dir=BENCHMARK_DIR,
        lp_max_weight=2,
        experiment_class=IncrementalExperiment,
        test_domain=domain,
        instances=['training01.pddl',
                   'training02.pddl',
                   'training03.pddl',
                   'training04.pddl',
                   'training05.pddl',
                   'training06.pddl',
                   'training09.pddl',
                   'training10.pddl',
                   'training11.pddl',],
        test_instances=["problem02-full.pddl",
                        "problem03-full.pddl",
                        "problem04-full.pddl",
                        "problem05-full.pddl",
                        "problem06-full.pddl",
                        "problem07-full.pddl",
                        "problem08-full.pddl",
                        "problem09-full.pddl",
                        "problem10-full.pddl",
                        "problem11-full.pddl",
                        # 'p-1-10.pddl',
                        # 'p-1-11.pddl',
                        # 'p-1-12.pddl',
                        # 'p-1-13.pddl',
                        # 'p-1-14.pddl',
                        # 'p-1-15.pddl',
                        # 'p-1-16.pddl',
                        # 'p-1-17.pddl',
                        # 'p-1-18.pddl',
                        # 'p-1-5.pddl',
                        # 'p-1-6.pddl',
                        # 'p-1-7.pddl',
                        # 'p-1-8.pddl',
                        # 'p-1-9.pddl',
        ],
        num_states=10000,
        initial_sample_size=100,
        distance_feature_max_complexity=5,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=10,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    visitall_incremental = dict(
        benchmark_dir=BENCHMARK_DIR,
        lp_max_weight=100,
        experiment_class=IncrementalExperiment,
        test_domain=domain,
        instances=['training01.pddl',
                   'training02.pddl',
                   'training03.pddl',
                   'training04.pddl',
                   'training05.pddl',
                   'training06.pddl',
                   'training09.pddl',
                   'training10.pddl',
                   'training11.pddl',],
        test_instances=["problem02-full.pddl",
                        "problem02-half.pddl",
                        "problem03-full.pddl",
                        "problem03-half.pddl",
                        'problem02-full.pddl',
                        'problem02-half.pddl',
                        'problem03-full.pddl',
                        'problem03-half.pddl',
                        'problem04-full.pddl',
                        'problem04-half.pddl',
                        'problem05-full.pddl',
                        'problem05-half.pddl',
                        'problem06-full.pddl',
                        'problem06-half.pddl',
                        'problem07-full.pddl',
                        'problem07-half.pddl',
                        'problem08-full.pddl',
                        'problem08-half.pddl',
                        'problem09-full.pddl',
                        'problem09-half.pddl',
                        'problem10-full.pddl',
                        'problem10-half.pddl',
                        'problem11-full.pddl',
                        'problem11-half.pddl',
                        "problem04-full.pddl",
        ],
        num_states=35000,
        initial_sample_size=100,
        distance_feature_max_complexity=10,
        max_concept_grammar_iterations=3,
        initial_concept_bound=10, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=10,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    parameters = {
        "problem02full": problem02full,
        "problem03full": problem03full,
        "problem03full_incremental": problem03full_incremental,
        "visitall_incremental": visitall_incremental,
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
