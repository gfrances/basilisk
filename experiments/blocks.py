#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR
from sltp.util.misc import update_dict

def experiments():
    domain = "domain.pddl"
    base = dict(
        domain_dir="blocks",
        domain=domain,
        benchmark_dir = PYPERPLAN_BENCHMARK_DIR
    )
    exps = dict()

    exps['learn'] = update_dict(
        base,
        test_domain=domain,
        instances=['task01.pddl',
                   'task02.pddl',
                   'task03.pddl',
                   'task04.pddl',
                   'task05.pddl',
                   'task06.pddl',
        ],
        test_instances=['task07.pddl',
                        'task08.pddl'],
        #distance_feature_max_complexity=5,
        num_states=100000,
        num_sampled_states=None,
        max_concept_size=7,
        concept_generator=None,
        parameter_generator=None
    )

    return exps


def generate_chosen_concepts(lang):
    """  """
    return [], [], []  # atoms, concepts, roles


def add_domain_parameters(language):
    return []
