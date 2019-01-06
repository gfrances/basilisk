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
    domain_dir = "visitall-opt11"
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
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=["problem03-full.pddl"],
        test_instances=["problem06-full.pddl"],
        test_domain=domain,
        distance_feature_max_complexity=10,
        num_states=1000, num_sampled_states=None, random_seed=12,
        max_concept_size=8, max_concept_grammar_iterations=3,
        concept_generator=None, parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,)

    # Incremental version
    problem03full_incremental = dict(
        benchmark_dir=BENCHMARK_DIR,
        lp_max_weight=10,
        experiment_class=IncrementalExperiment,
        test_domain=domain,
        instances="problem03-full.pddl",
        test_instances=["problem09-full.pddl"],
        num_states=500,
        initial_sample_size=8,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=5,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )


    parameters = {
        "problem02full": problem02full,
        "problem03full": problem03full,
        "problem03full_incremental": problem03full_incremental,
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
