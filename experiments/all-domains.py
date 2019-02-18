#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, DOWNWARD_BENCHMARKS_DIR
from basilisk.incremental import IncrementalExperiment

from sltp.util.misc import update_dict
from defaults import generate_experiment
from tarski.dl import PrimitiveRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept

def experiment():
    benchmark_dir = DOWNWARD_BENCHMARKS_DIR

    experiments = dict()
    list_of_experiments = []

    list_of_domains = domains #[x for x in os.listdir(benchmark_dir)]
    for domain in list_of_domains:
        domain_dir = os.path.join(DOWNWARD_BENCHMARKS_DIR, domain)
        if not os.path.isdir(domain_dir):
            continue
        if "domain.pddl" not in os.listdir(domain_dir):
            continue

        instances = []
        experiment_name = "experiment_"+domain
        for problem in os.listdir(domain_dir):
            if problem == "domain.pddl":
                continue
            instances.append(problem)

        experiments[experiment_name] = dict(
            lp_max_weight=5,
            experiment_class=IncrementalExperiment,
            benchmark_dir=benchmark_dir,
            instances=[instances[0]],
            test_instances=[instances[2]],
            test_domain='domain.pddl',
            # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
            # in batches, so we can set them high enough.
            num_states=12000,
            initial_sample_size=50,
            max_concept_grammar_iterations=3,
            initial_concept_bound=7, max_concept_bound=12, concept_bound_step=2,
            batch_refinement_size=10,
            clean_workspace=False,
            parameter_generator=add_domain_parameters,
        )

        # Select the actual experiment parameters according to the command-line option
        parameters = experiments[experiment_name]
        parameters["domain_dir"] = parameters.get("domain_dir", domain_dir)
        parameters["domain"] = parameters.get("domain", 'domain.pddl')
        list_of_experiments.append(generate_experiment(**parameters))

    return list_of_experiments

def add_domain_parameters(language):
    return []


if __name__ == "__main__":
    list_of_exp = experiment()
    for exp in list_of_exp:
        try:
            exp.run(sys.argv[1:])
        except ValueError:
            pass
