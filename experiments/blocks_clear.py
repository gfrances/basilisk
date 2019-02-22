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
    domain_dir = "blocks"

    exps = dict()

    exps["clear_5"] = dict(
        instances="instance_5_clear_x_1.pddl",
        test_domain=domain,
        test_instances=[
            "instance_5_clear_x_2.pddl",
            "instance_8_clear_x_0.pddl"
        ],
        lp_max_weight=5,
        num_states=2000, max_width=[-1],
        num_sampled_states=300,
        max_concept_size=6,
        max_concept_grammar_iterations=None,
    )

    parameters = exps[experiment_name]
    parameters["domain_dir"] = parameters.get("domain_dir", domain_dir)
    parameters["domain"] = parameters.get("domain", domain)
    return generate_experiment(**parameters)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
