#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR, DOWNWARD_BENCHMARKS_DIR
from basilisk.incremental import IncrementalExperiment

from sltp.util.misc import update_dict
from defaults import generate_experiment
from tarski.dl import PrimitiveRole, NominalConcept, ExistsConcept, NotConcept, UniversalConcept, AndConcept, \
    ForallConcept, EmptyConcept, GoalRole, GoalConcept, PrimitiveConcept, NullaryAtom, StarRole, \
    Concept, EqualConcept, InverseRole


def experiment(experiment_name):
    domain = "domain.pddl"
    domain_dir = "gripper-m"
    benchmark_dir = BENCHMARK_DIR

    experiments = dict()
    experiments["prob01"] = dict(
        lp_max_weight=10,
        benchmark_dir=benchmark_dir,
        instances=['test01.pddl',
                   'test02.pddl',
                   'test03.pddl',
                   'test04.pddl',
                   'test05.pddl',
                   'test06.pddl',
                   "prob02.pddl",],
        test_instances=["prob01.pddl",
                        "prob03.pddl",
                        "prob04.pddl",
                        "prob05.pddl",
                        "prob06.pddl"],
        test_domain=domain,
        # instances="task01.pddl",
        num_states=5000,  # num_sampled_states=None,  random_seed=12,
        max_concept_size=2, max_concept_grammar_iterations=3,
        concept_generator=None,
        # concept_generator=generate_chosen_concepts,
        # parameter_generator=add_domain_parameters,
        parameter_generator=None,
        feature_namer=feature_namer,
    )

    # Experiment used in the paper
    experiments["gripper_std_inc"] = dict(
        lp_max_weight=5,
        experiment_class=IncrementalExperiment,
        instances=['test01.pddl',
                   'test02.pddl',
                   'test03.pddl',
                   'test04.pddl',
                   'test05.pddl',
                   'test06.pddl',
                   'prob03.pddl',
                   'prob_3balls_3rooms_1rob.pddl'],
        test_instances=["prob01.pddl",
                        "prob02.pddl",
                        "prob03.pddl",
                        "prob04.pddl",
                        "prob05.pddl",
                        "prob06.pddl",
                        'prob_3balls_3rooms_1rob.pddl',
                        "prob_4balls_4rooms_1rob.pddl",
                        "prob_4balls_4rooms_2rob.pddl",
                        "prob_10balls_4rooms_1rob.pddl",],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=12000,
        initial_sample_size=100,
        max_concept_grammar_iterations=None, random_seed=19,
        initial_concept_bound=8, max_concept_bound=12, concept_bound_step=2,
        batch_refinement_size=50,
        clean_workspace=False,
        parameter_generator=None, #add_domain_parameters,
        feature_namer=feature_namer,
    )

    experiments["guillem"] = dict(
        lp_max_weight=10,
        experiment_class=IncrementalExperiment,
        instances=[
            'prob_4balls_2rooms.pddl',
            'prob_3balls_3rooms_1rob.pddl'],
        test_instances=["prob01.pddl",
                        "prob02.pddl",
                        "prob03.pddl",
                        "prob04.pddl",
                        "prob05.pddl",
                        "prob06.pddl",
                        'prob_3balls_3rooms_1rob.pddl',
                        "prob_4balls_4rooms_1rob.pddl",
                        "prob_4balls_4rooms_2rob.pddl",
                        "prob_10balls_4rooms_1rob.pddl",],
        test_domain=domain,
        # This is number of sampled states *per training instance*. In an increm. experiment, they will be processed
        # in batches, so we can set them high enough.
        num_states=12000,
        initial_sample_size=20,
        max_concept_grammar_iterations=None,
        initial_concept_bound=8, max_concept_bound=10, concept_bound_step=2,
        batch_refinement_size=10,
        clean_workspace=False,
        parameter_generator=None, #add_domain_parameters,
        feature_namer=feature_namer,
    )


    experiments["original"] = dict(
        lp_max_weight=10,
        benchmark_dir=BENCHMARK_DIR,
        instances=['prob01.pddl',
                   'prob02.pddl',
                   ],
        test_instances=[
                        "prob04.pddl",
                        "prob05.pddl",
                        "prob06.pddl"
        ],
        num_states=2000, max_width=[-1],
        num_sampled_states=100,
        test_domain=domain,
        max_concept_size=8, max_concept_grammar_iterations=3,
        concept_generator=None,
        # parameter_generator=add_domain_parameters,
        parameter_generator=None,
        feature_namer=feature_namer,
    )

    experiments["several_rooms"] = dict(
        lp_max_weight=10,
        benchmark_dir=BENCHMARK_DIR,
        instances=["prob_3balls_3rooms_1rob.pddl"],
        num_states=100000,
        num_sampled_states=10000,
        test_instances=["prob02.pddl"],
        test_domain=domain,
        # random_seed=12,
        max_concept_size=8, max_concept_grammar_iterations=3,
        concept_generator=None,
        # concept_generator=generate_chosen_concepts,
        parameter_generator=add_domain_parameters,
        feature_namer=feature_namer,
    )

    experiments["2rooms"] = dict(
        lp_max_weight=10,
        benchmark_dir=BENCHMARK_DIR,
        instances=["small.pddl"],
        num_states=100000,
        num_sampled_states=10000,
        test_instances=["small-test.pddl"],
        test_domain=domain,
        # random_seed=12,
        max_concept_size=8, max_concept_grammar_iterations=3,
        concept_generator=None,
        # concept_generator=generate_chosen_concepts,
        parameter_generator=add_domain_parameters,
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

    concepts = [c1, c2, rx, c3, c4]
    return [], concepts, []  # atoms, concepts, roles

def generate_chosen_concepts_without_nominal(lang):
    """  """
    # card[Exists(at,Not({roomb}))] 4    C1
    # card[Exists(carry,<universe>)] 2   C2
    # bool[Exists(at-robby,{roomb})] 3   RX
    # card[Exists(gripper,Exists(at-robby,{roomb}))] 5   (intermediate)
    # card[Exists(carry,Exists(gripper,Exists(at-robby,{roomb})))] 7

    obj_t = lang.Object

    universal = UniversalConcept("object")

    at = PrimitiveRole(lang.get("at"))
    at_g = GoalRole(lang.get("at"))
    carry = PrimitiveRole(lang.get("carry"))
    at_robby = PrimitiveRole(lang.get("at-robby"))
    gripper = PrimitiveRole(lang.get("gripper"))
    at_g_at = EqualConcept(at_g, at, 'object')


    nominal_subst = ExistsConcept(InverseRole(at_g), universal)
    g1 = ExistsConcept(at, NotConcept(nominal_subst, obj_t))
    g2 = ExistsConcept(carry, UniversalConcept("object"))
    g3 = ExistsConcept(carry, ExistsConcept(gripper, ExistsConcept(at_robby, nominal_subst)))
    g4 = ExistsConcept(at_robby, AndConcept(NotConcept(nominal_subst, obj_t), ExistsConcept(InverseRole(at), universal), 'object'))

    concepts = [nominal_subst, g1, g2, g3, g4]
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
        "card[Forall(at,And(Forall(Inverse(at-robby),<empty>),Not({roomb})))]": "nballs-either-held-or-in-a-room-!=B-with-no-robot",
        "card[Forall(at,{roomb})]": "nballs-being-carried-or-in-B",
        "card[Forall(carry,Exists(gripper,Exists(at-robby,{roomb})))]": "nballs-either-not-carried-or-in-room-B"
        # "": "",
    }.get(s, s)


if __name__ == "__main__":
    exp = experiment(sys.argv[1])
    exp.run(sys.argv[2:])
