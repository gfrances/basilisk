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
    domain_dir = "gripper-m"
    benchmark_dir = BENCHMARK_DIR

    experiments = dict()
    experiments["prob01"] = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=["prob01.pddl", "prob02.pddl", "prob03.pddl"],
        test_instances=["prob04.pddl",
                        "prob05.pddl",
                        "prob06.pddl"],
        test_domain=domain,
        # instances="task01.pddl",
        num_states=300,  # num_sampled_states=None,  random_seed=12,
        max_concept_size=10, max_concept_grammar_iterations=3,
        concept_generator=None,
        # concept_generator=generate_chosen_concepts,
        # parameter_generator=add_domain_parameters,
        parameter_generator=None,
        feature_namer=feature_namer,
    )

    experiments["several_rooms"] = update_dict(
        experiments["prob01"],
        num_states=800,
        instances=["prob_3balls_3rooms_1rob.pddl"],
    )

    # An experiment with the original gripper domain,
    # which has some spureous difficulties to generate the right concepts
    experiments["gripper_original"] = dict(
        domain_dir="gripper",
        benchmark_dir=PYPERPLAN_BENCHMARK_DIR,
        lp_max_weight=10,
        experiment_class=IncrementalExperiment,
        instances=["task01.pddl", "task02.pddl", "task03.pddl"],
        test_instances=["task04.pddl", "task05.pddl", "task06.pddl"],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=500,
        initial_sample_size=20,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=5,
        # quiet=True,
        clean_workspace=False,
        # concept_generator=generate_chosen_concepts,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    experiments["gripper_std_inc"] = dict(
        # domain_dir="blocks-downward",
        lp_max_weight=10,
        experiment_class=IncrementalExperiment,
        instances=["prob01.pddl", "prob02.pddl", "prob03.pddl"],
        test_instances=["prob04.pddl", "prob05.pddl", "prob06.pddl"],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=500,
        initial_sample_size=20,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=8, concept_bound_step=2,
        batch_refinement_size=5,
        # quiet=True,
        clean_workspace=False,
        # concept_generator=generate_chosen_concepts,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    experiments["gripper_gen_inc"] = update_dict(
        experiments["gripper_std_inc"],
        instances=["prob03.pddl", "prob_3balls_3rooms_1rob.pddl"],
        test_instances=["prob_4balls_4rooms_1rob.pddl"],
        initial_sample_size=50,
        max_concept_grammar_iterations=3,
        initial_concept_bound=8, max_concept_bound=8, concept_bound_step=2,
        batch_refinement_size=5,
        # quiet=True,
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
