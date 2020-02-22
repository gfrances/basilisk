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
    # benchmark_dir = BENCHMARK_DIR
    benchmark_dir = PYPERPLAN_BENCHMARK_DIR
    # domain_dir = "gripper"

    experiments = dict()
    experiments["task01_inc"] = dict(
        lp_max_weight=5,
        benchmark_dir=benchmark_dir,
        experiment_class=IncrementalExperiment,
        instances=['task01.pddl',
                   'task02.pddl',
                   'task03.pddl',
                   'task04.pddl',
                   'task05.pddl',
        ],
        test_instances=[
        'task06.pddl',
        'task07.pddl',
        'task08.pddl',
        ],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=12000,
        initial_sample_size=125,
        max_concept_grammar_iterations=3,
        initial_concept_bound=12, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=10,
        clean_workspace=False,
        parameter_generator=None,
        concept_generator=generate_chosen_concepts,
        feature_namer=feature_namer,
    )

    experiments['learn'] = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=['task01.pddl',
                   'task02.pddl',
                   'task03.pddl',
                   'task04.pddl',
                   'task05.pddl',
                   'task06.pddl',
        ],
        test_instances=[

                        'task07.pddl',
                        'task08.pddl',
        ],
        test_domain=domain,
        distance_feature_max_complexity=5,
        num_states=1000000, num_sampled_states=None, random_seed=12,
        max_concept_size=10, max_concept_grammar_iterations=5,
        concept_generator=None, parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,)


    experiments["handcraft_learn"] = dict(
        lp_max_weight=5,
        benchmark_dir=benchmark_dir,
        instances=['task01.pddl',
                   'task02.pddl',
                   'task03.pddl',
                   'task04.pddl',
                   'task05.pddl',
                   'task06.pddl',
        ],
        test_instances=[
        'task06.pddl',
        'task07.pddl',
        'task08.pddl',
        ],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=1000000,
        distance_feature_max_complexity=5,
        max_concept_grammar_iterations=3,
        initial_concept_bound=12, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=10,
        clean_workspace=False,
        parameter_generator=None,
        concept_generator=generate_chosen_concepts,
        feature_namer=feature_namer,
    )

    parameters = experiments[experiment_name]
    parameters["domain_dir"] = parameters.get("domain_dir", domain_dir)
    parameters["domain"] = parameters.get("domain", domain)
    return generate_experiment(**parameters)


def add_domain_parameters(language):
    return []

def generate_chosen_concepts(lang):
    """  """
    # card[Exists(at,Not({roomb}))] 4    C1
    # card[Exists(carry,<universe>)] 2   C2
    # bool[Exists(at-robby,{roomb})] 3   RX
    # card[Exists(gripper,Exists(at-robby,{roomb}))] 5   (intermediate)
    # card[Exists(carry,Exists(gripper,Exists(at-robby,{roomb})))] 7

    obj_t = lang.Object
    block_t = lang.get('block')

    handempty = NullaryAtom(lang.get("handempty"))
    on = PrimitiveRole(lang.get("on"))
    on_g = GoalRole(lang.get("on"))
    ontable = PrimitiveConcept(lang.get("ontable"))
    clear = PrimitiveConcept(lang.get("clear"))
    block = PrimitiveConcept(lang.get("block"))

    on_g_on = EqualConcept(on_g, on, 'block')
    M_t = AndConcept(on_g_on, ontable, 'block')
    W = ForallConcept(on_g, ForallConcept(StarRole(on_g), on_g_on))
    M_t_inv = AndConcept(NotConcept(W, obj_t), NotConcept(ontable, obj_t), 'block')

    concepts = [M_t, W, M_t_inv]
    return [], concepts, []  # atoms, concepts, roles

def feature_namer(feature):
    s = str(feature)
    return {
    }.get(s, s)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
