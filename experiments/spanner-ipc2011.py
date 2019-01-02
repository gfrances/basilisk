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
    domain_dir = "spanner-ipc2011"
    benchmark_dir = BENCHMARK_DIR

    spanner_1 = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        #instances=["pfile01-001-training-inst.pddl"],
        instances=["prob-3-3-3-1540903410.pddl"],
                   #"prob-4-4-3-1540907456.pddl",
                   #"prob-4-3-3-1540907466.pddl",],
                   #"prob-10-10-10-1540903568.pddl",
                   #"prob-15-10-8-1540913795.pddl"],
        test_instances=["pfile01-001.pddl",
                        "pfile01-002.pddl",
                        "pfile01-003.pddl",
                        "pfile01-004.pddl",
                        "pfile01-005.pddl",
                        "pfile02-006.pddl",
                        "pfile02-007.pddl",
                        "pfile02-008.pddl",
                        "pfile02-009.pddl",
                        "pfile02-010.pddl",
                        "pfile03-011.pddl",
                        "pfile03-012.pddl",
                        "pfile03-013.pddl",
                        "pfile03-014.pddl",
                        "pfile03-015.pddl",
                        "pfile04-016.pddl",
                        "pfile04-017.pddl",
                        "pfile04-018.pddl",
                        "pfile04-019.pddl",
                        "pfile04-020.pddl",
                        "pfile05-021.pddl",
                        "pfile05-022.pddl",
                        "pfile05-023.pddl",
                        "pfile05-024.pddl",
                        "pfile05-025.pddl",
                        "pfile06-026.pddl",
                        "pfile06-027.pddl",
                        "pfile06-028.pddl",
                        "pfile06-029.pddl",
                        "pfile06-030.pddl",],
        test_domain=domain,
        distance_feature_max_complexity=0,
        num_states=300, num_sampled_states=None, random_seed=12,
        max_concept_size=10, max_concept_grammar_iterations=3,
        concept_generator=None,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    spanner_1_incremental = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=["pfile01-001-training-inst.pddl"],
        test_instances=["pfile01-001.pddl",
                        "pfile01-002.pddl",
                        "pfile01-003.pddl",
                        "pfile01-004.pddl",
                        "pfile01-005.pddl",
                        "pfile02-006.pddl",
                        "pfile02-007.pddl",
                        "pfile02-008.pddl",
                        "pfile02-009.pddl",
                        "pfile02-010.pddl",
                        "pfile03-011.pddl",
                        "pfile03-012.pddl",
                        "pfile03-013.pddl",
                        "pfile03-014.pddl",
                        "pfile03-015.pddl",
                        "pfile04-016.pddl",
                        "pfile04-017.pddl",
                        "pfile04-018.pddl",
                        "pfile04-019.pddl",
                        "pfile04-020.pddl",
                        "pfile05-021.pddl",
                        "pfile05-022.pddl",
                        "pfile05-023.pddl",
                        "pfile05-024.pddl",
                        "pfile05-025.pddl",
                        "pfile06-026.pddl",
                        "pfile06-027.pddl",
                        "pfile06-028.pddl",
                        "pfile06-029.pddl",
                        "pfile06-030.pddl",],
        test_domain=domain,
        distance_feature_max_complexity=0,
        num_states=100,
        initial_sample_size=8,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=5,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    parameters = {
        "spanner_1": spanner_1,
        "spanner_1_incremental": spanner_1_incremental,
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
