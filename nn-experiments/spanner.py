#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from sltp.util.misc import update_dict

def experiments():
    domain = "domain.pddl"
    base = dict(
        domain_dir="spanner",
        domain=domain,
    )
    exps = dict()

    exps['learn'] = update_dict(
        base,
        instances=['training-2-1-5.pddl',
                   'training-2-2-5.pddl',
                   'training-3-1-5.pddl',
                   'training-3-3-5.pddl',
                   'training-4-2-5.pddl',
                   'training-4-4-5.pddl',
                   'training-5-3-5.pddl',
                   'training-5-4-5.pddl',
                   'training-5-5-5.pddl',
                   'training-1-1-6.pddl',
                   'training-1-1-20.pddl',],
        test_instances=["prob-3-3-3-1540903410.pddl",
                        "prob-4-3-3-1540907466.pddl",
                        "prob-4-3-3-1540907466.pddl",
                        "prob-10-10-10-1540903568.pddl",
                        "prob-15-10-8-1540913795.pddl"],
        test_domain=domain,
        #distance_feature_max_complexity=5,
        num_tested_states=10000,
        num_states=10000, max_width=[-1],
        num_sampled_states=None,
        max_concept_size=8, max_concept_grammar_iterations=3,
        concept_generator=None,
        parameter_generator=None
    )

    return exps


def generate_chosen_concepts(lang):
    """  """
    return [], [], []  # atoms, concepts, roles


def add_domain_parameters(language):
    return []
