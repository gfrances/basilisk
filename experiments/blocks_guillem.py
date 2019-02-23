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

    exps["on_x_y_big"] = dict(
        instances=[
            "inst_on_x_y_10.pddl",
            "inst_on_x_y_14.pddl",
            "holding_a_b_unclear.pddl",
        ],
        test_domain=domain,
        test_instances=[
            "inst_on_x_y_5.pddl",
            "inst_on_x_y_6.pddl",
            "inst_on_x_y_7.pddl",
            "inst_on_x_y_7_2.pddl",
        ],
        lp_max_weight=5,
        num_states=2000, max_width=[-1],
        num_sampled_states=50,
        max_concept_size=6,
        max_concept_grammar_iterations=None,
    )

    exps["on_x_y_inc"] = dict(
        lp_max_weight=5,
        experiment_class=IncrementalExperiment,
        instances=[
            "inst_on_x_y_10.pddl",
        ],
        test_instances=[
            "inst_on_x_y_11.pddl",
        ],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=12000,
        initial_sample_size=50,
        initial_concept_bound=6, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=10,
        clean_workspace=False,
    )

    parameters = exps[experiment_name]
    parameters["domain_dir"] = parameters.get("domain_dir", domain_dir)
    parameters["domain"] = parameters.get("domain", domain)
    return generate_experiment(**parameters)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
