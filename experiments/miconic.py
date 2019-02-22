#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR
from basilisk.incremental import IncrementalExperiment

from sltp.util.misc import update_dict
from defaults import generate_experiment
from tarski.dl import (
    PrimitiveRole,
    NominalConcept,
    ExistsConcept,
    NotConcept,
    UniversalConcept,
    AndConcept,
    ForallConcept,
    EmptyConcept,
    PrimitiveConcept,
    Concept
)


def experiment(experiment_name=None):
    domain = "domain.pddl"
    domain_dir = "miconic"
    benchmark_dir = BENCHMARK_DIR

    miconic_1_simple = dict(
        experiment_class=IncrementalExperiment,
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=[
            's1-0.pddl',
            's1-1.pddl',
            's1-2.pddl',
            's1-3.pddl',
            # 's1-4.pddl',
            's2-0.pddl',
            's2-1.pddl',
            's2-2.pddl',
            's2-3.pddl',
            # 's2-4.pddl',
            # 's3-0.pddl',
            # 's3-1.pddl',
            # 's3-2.pddl',
            # 's3-3.pddl',
            # 's3-4.pddl',
            # 's4-0.pddl',
            # 's4-1.pddl',
            # 's4-2.pddl',
            # 's4-3.pddl',
            # 's4-4.pddl',
            ],
        test_instances=['s1-4.pddl', 's2-4.pddl', 's3-4.pddl'],
        test_domain=domain,
        # distance_feature_max_complexity=10,
        num_states=20000,
        initial_sample_size=50,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=16, concept_bound_step=2,
        batch_refinement_size=50,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
        concept_generator=generate_chosen_concepts,
    )

    miconic_1_incremental = dict(
        experiment_class=IncrementalExperiment,
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=[
            's1-0.pddl',
            's1-1.pddl',
            's1-2.pddl',
            's1-3.pddl',
            's1-4.pddl',
            's2-0.pddl',
            's2-1.pddl',
            's2-2.pddl',
            's2-3.pddl',
            's2-4.pddl',
            's3-0.pddl',
            's3-1.pddl',
            's3-2.pddl',
            's3-3.pddl',
            's3-4.pddl',
            's4-0.pddl',
            's4-1.pddl',
            's4-2.pddl',
            's4-3.pddl',
            's4-4.pddl',
            ],
        test_instances=['s5-0.pddl'],
        test_domain=domain,
        # distance_feature_max_complexity=10,
        num_states=20000,
        initial_sample_size=50,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=16, concept_bound_step=2,
        batch_refinement_size=50,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    parameters = {
        "miconic_1_incremental": miconic_1_incremental,
        "miconic_1_simple": miconic_1_simple,
    }.get(experiment_name or "test")

    return generate_experiment(domain_dir, domain, **parameters)


def generate_chosen_concepts(lang):
    """  """

    obj_t = lang.Object

    lift_at = PrimitiveConcept(lang.get("lift-at"))
    boarded = PrimitiveConcept(lang.get("boarded"))
    served = PrimitiveConcept(lang.get("served"))

    origin = PrimitiveRole(lang.get("origin"))
    destin = PrimitiveRole(lang.get("destin"))

    M1 = AndConcept(NotConcept(boarded, obj_t), NotConcept(served, obj_t),
                    "object")
    M2 = AndConcept(boarded, NotConcept(served, obj_t), "object")
    M3 = ExistsConcept(origin, lift_at)
    M4 = ExistsConcept(destin, lift_at)

    F1 = AndConcept(M1, M3, "object")
    F2 = AndConcept(M1, NotConcept(M3, obj_t), "object")
    F3 = AndConcept(M2, M4, "object")
    F4 = AndConcept(M2, NotConcept(M4, obj_t), "object")
    return [], [F1, F2, F3, F4], []  # atoms, concepts, roles


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
