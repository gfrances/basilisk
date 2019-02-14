#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR
from basilisk.incremental import IncrementalExperiment

from sltp.util.misc import update_dict
from defaults import generate_experiment
from tarski.dl import PrimitiveRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept


def experiment(experiment_name):
    domain = "domain.pddl"
    domain_dir = "gripper"
    benchmark_dir = PYPERPLAN_BENCHMARK_DIR

    experiments = dict()

    experiments["incremental"] = dict(
        lp_max_weight=10,
        experiment_class=IncrementalExperiment,
        benchmark_dir=benchmark_dir,
        instances=['task01.pddl',
                   'task02.pddl',
                   'task03.pddl',
                   'task04.pddl',],
        test_instances=['task05.pddl',
                        'task06.pddl',
                        'task07.pddl',
                        'task08.pddl',
                        'task09.pddl',
                        'task10.pddl',
                        'task11.pddl',
                        'task12.pddl',
                        'task13.pddl',
                        'task14.pddl',
                        'task15.pddl',
                        'task16.pddl',
                        'task17.pddl',
                        'task18.pddl',
                        'task19.pddl',
                        'task20.pddl',],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=3000,
        initial_sample_size=100,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=10,
        clean_workspace=False,
        parameter_generator=None,
        feature_namer=feature_namer,
    )

    # Select the actual experiment parameters according to the command-line option
    parameters = experiments[experiment_name]
    parameters["domain_dir"] = parameters.get("domain_dir", domain_dir)
    parameters["domain"] = parameters.get("domain", domain)
    return generate_experiment(**parameters)


def generate_chosen_concepts(lang):
    """  """
    # card[Exists(at,Not({roomb}))] 4    C1
    # card[Exists(carry,<universe>)] 2   C2
    # bool[Exists(at-robby,{roomb})] 3   RX
    # card[Exists(gripper,Exists(at-robby,{roomb}))] 5   (intermediate)
    # card[Exists(carry,Exists(gripper,Exists(at-robby,{roomb})))] 7

    obj_t = lang.Object

    at = PrimitiveRole(lang.get("at"))
    carry = PrimitiveRole(lang.get("carry"))
    at_robby = PrimitiveRole(lang.get("at-robby"))
    gripper = PrimitiveRole(lang.get("gripper"))
    x_param = NominalConcept("roomb", obj_t)
    c1 = ExistsConcept(at, NotConcept(x_param, obj_t))
    c2 = ExistsConcept(carry, UniversalConcept("object"))
    rx = ExistsConcept(at_robby, x_param)
    c3 = ExistsConcept(gripper, rx)
    c4 = ExistsConcept(carry, c3)

    concepts = [c1, c2, c3, c4]
    return [], concepts, []  # atoms, concepts, roles


def debug_weird_concept(lang):
    #  card[And(Forall(carry,<empty>), Forall(at-robby,{roomb}))]

    obj_t = lang.Object
    bot = EmptyConcept("object")

    carry = PrimitiveRole(lang.get("carry"))
    at_robby = PrimitiveRole(lang.get("at-robby"))
    x_param = NominalConcept("roomb", obj_t)

    c1 = ForallConcept(carry, bot)
    c2 = ForallConcept(at_robby, x_param)
    c = AndConcept(c1, c2, "object")

    concepts = [c]
    return [], concepts, []  # atoms, concepts, roles


def add_domain_parameters(language):
    return [language.constant("roomb", "object")]


def feature_namer(feature):
    s = str(feature)
    return {
        "card[free]": "nfree-grippers",
        "bool[Exists(at-robby,{roomb})]": "robby-is-at-B",
        "card[Exists(at,Exists(Inverse(at-robby),<universe>))]": "nballs-in-room-with-some-robot",
        "card[And(Exists(gripper,Exists(at-robby,{roomb})),free)]": "nfree-grippers-at-B",
        "card[Exists(at,{roomb})]": "nballs-at-B",
        "card[Exists(at,Not({roomb}))]": "nballs-not-at-B",
        "card[Exists(carry,<universe>)]": "nballs-carried",
        "card[Exists(at-robby,{roomb})]": "nrobots-at-B",
        "card[Exists(gripper,Exists(at-robby,{roomb}))]": "ngrippers-at-B",
        "card[Exists(carry,Exists(gripper,Exists(at-robby,{roomb})))]": "nballs-carried-in-B",
        "card[Exists(at,And(Forall(Inverse(at-robby),<empty>), Not({roomb})))]": "nballs-in-some-room-notB-without-any-robot",
        # "": "",
    }.get(s, s)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
