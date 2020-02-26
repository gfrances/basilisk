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
        initial_sample_size=100,
        max_concept_grammar_iterations=10,
        initial_concept_bound=8, max_concept_bound=16, concept_bound_step=2,
        batch_refinement_size=50,
        clean_workspace=False,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
        concept_generator=generate_chosen_concepts,
    )

    # Experiment used in the paper
    miconic_1_incremental = dict(
        experiment_class=IncrementalExperiment,
        lp_max_weight=4,
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
        ],
        test_instances=['s5-0.pddl',
                        's5-1.pddl',
                        's5-2.pddl',
                        's5-3.pddl',
                        's5-4.pddl',
                        's6-0.pddl',
                        's6-1.pddl',
                        's6-2.pddl',
                        's6-3.pddl',
                        's6-4.pddl',
                        's7-0.pddl',
                        's7-1.pddl',
                        's7-2.pddl',
                        's7-3.pddl',
                        's7-4.pddl',
                        's8-0.pddl',
                        's8-1.pddl',
                        's8-2.pddl',
                        's8-3.pddl',
                        's8-4.pddl',
                        's9-0.pddl',
                        's9-1.pddl',
                        's9-2.pddl',
                        's9-3.pddl',
                        's9-4.pddl',
                        's10-0.pddl',
                        's10-1.pddl',
                        's10-2.pddl',
                        's10-3.pddl',
                        's10-4.pddl',
                        's1-0.pddl',
                        's11-0.pddl',
                        's11-1.pddl',
                        's11-2.pddl',
                        's11-3.pddl',
                        's11-4.pddl',
                        's1-1.pddl',
                        's12-0.pddl',
                        's12-1.pddl',
                        's12-2.pddl',
                        's12-3.pddl',
                        's12-4.pddl',
                        's1-2.pddl',
                        's13-0.pddl',
                        's13-1.pddl',
                        's13-2.pddl',
                        's13-3.pddl',
                        's13-4.pddl',
                        's1-3.pddl',
                        's14-0.pddl',
                        's14-1.pddl',
                        's14-2.pddl',
                        's14-3.pddl',
                        's14-4.pddl',
                        's1-4.pddl',
                        's15-0.pddl',
                        's15-1.pddl',
                        's15-2.pddl',
                        's15-3.pddl',
                        's15-4.pddl',
                        's16-0.pddl',
                        's16-1.pddl',
                        's16-2.pddl',
                        's16-3.pddl',
                        's16-4.pddl',
                        's17-0.pddl',
                        's17-1.pddl',
                        's17-2.pddl',
                        's17-3.pddl',
                        's17-4.pddl',
                        's18-0.pddl',
                        's18-1.pddl',
                        's18-2.pddl',
                        's18-3.pddl',
                        's18-4.pddl',
                        's19-0.pddl',
                        's19-1.pddl',
                        's19-2.pddl',
                        's19-3.pddl',
                        's19-4.pddl',
                        's20-0.pddl',
                        's20-1.pddl',
                        's20-2.pddl',
                        's20-3.pddl',
                        's20-4.pddl',
                        's2-0.pddl',
                        's21-0.pddl',
                        's21-1.pddl',
                        's21-2.pddl',
                        's21-3.pddl',
                        's21-4.pddl',
                        's2-1.pddl',
                        's22-0.pddl',
                        's22-1.pddl',
                        's22-2.pddl',
                        's22-3.pddl',
                        's22-4.pddl',
                        's2-2.pddl',
                        's23-0.pddl',
                        's23-1.pddl',
                        's23-2.pddl',
                        's23-3.pddl',
                        's23-4.pddl',
                        's2-3.pddl',
                        's24-0.pddl',
                        's24-1.pddl',
                        's24-2.pddl',
                        's24-3.pddl',
                        's24-4.pddl',
                        's2-4.pddl',
                        's25-0.pddl',
                        's25-1.pddl',
                        's25-2.pddl',
                        's25-3.pddl',
                        's25-4.pddl',
                        's26-0.pddl',
                        's26-1.pddl',
                        's26-2.pddl',
                        's26-3.pddl',
                        's26-4.pddl',
                        's27-0.pddl',
                        's27-1.pddl',
                        's27-2.pddl',
                        's27-3.pddl',
                        's27-4.pddl',
                        's28-0.pddl',
                        's28-1.pddl',
                        's28-2.pddl',
                        's28-3.pddl',
                        's28-4.pddl',
                        's29-0.pddl',
                        's29-1.pddl',
                        's29-2.pddl',
                        's29-3.pddl',
                        's29-4.pddl',
                        's30-0.pddl',
                        's30-1.pddl',
                        's30-2.pddl',
                        's30-3.pddl',
                        's30-4.pddl',
        ],
        test_domain=domain,
        #distance_feature_max_complexity=5,
        num_states=12000,
        initial_sample_size=50,
        max_concept_grammar_iterations=10,
        initial_concept_bound=8, max_concept_bound=16, concept_bound_step=2,
        batch_refinement_size=50,
        clean_workspace=False,
        parameter_generator=None,
        feature_namer=feature_namer,
        #concept_generator=generate_chosen_concepts,
    )

    parameters = {
        "miconic_1_incremental": miconic_1_incremental,
        "miconic_1_simple": miconic_1_simple,
    }.get(experiment_name or "test")

    return generate_experiment(domain_dir, domain, **parameters)


def generate_chosen_concepts(lang):
    """  """

    obj_t = lang.Object

    passenger = PrimitiveConcept(lang.get("passenger"))
    lift_at = PrimitiveConcept(lang.get("lift-at"))
    boarded = PrimitiveConcept(lang.get("boarded"))
    served = PrimitiveConcept(lang.get("served"))

    origin = PrimitiveRole(lang.get("origin"))
    destin = PrimitiveRole(lang.get("destin"))

    # M1 = AndConcept(NotConcept(boarded, obj_t), NotConcept(served, obj_t),
    #                 "object")
    # M2 = AndConcept(boarded, NotConcept(served, obj_t), "object") # K(m2) = 4
    M3 = ExistsConcept(origin, lift_at)
    M4 = ExistsConcept(destin, lift_at) # K(M4) = 3

    # F1 = AndConcept(M1, M3, "object")
    # F2 = AndConcept(M1, NotConcept(M3, obj_t), "object")
    # F3 = AndConcept(M2, M4, "object")
    # F4 = AndConcept(M2, NotConcept(M4, obj_t), "object")

    not_boarded_nor_served = AndConcept(NotConcept(boarded, obj_t), NotConcept(served, obj_t), "object")

    F1 = AndConcept(passenger, served, "object")
    F2 = AndConcept(passenger, AndConcept(boarded, M4, "object"), "object")
    F3 = AndConcept(passenger, AndConcept(boarded, ForallConcept(destin, NotConcept(lift_at, obj_t)), "object"), "object")
    F4 = AndConcept(passenger, AndConcept(not_boarded_nor_served, M3, "object"), "object")
    F5 = AndConcept(passenger, AndConcept(not_boarded_nor_served, ForallConcept(origin, NotConcept(lift_at, obj_t)), "object"), "object")

    return [], [F1, F2, F3, F4, F5], []  # atoms, concepts, roles


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
