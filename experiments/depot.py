from basilisk import PYPERPLAN_BENCHMARK_DIR
from sltp.util.misc import update_dict


def experiments():
    domain = "domain.pddl"
    base = dict(
        domain_dir="blocks",
        domain=domain,
        benchmark_dir=PYPERPLAN_BENCHMARK_DIR
    )
    exps = dict()

    exps['small'] = update_dict(
        base,
        test_domain=domain,
        instances=['task01.pddl',
                   'task02.pddl',
                   ],
        test_instances=[
            'task05.pddl',
            'task10.pddl',
            'task15.pddl',
            'task20.pddl',
        ],
        # distance_feature_max_complexity=5,
        num_states=100000,
        num_sampled_states=None,
        max_concept_size=8,
        concept_generation_timeout=600,  # in seconds
        concept_generator=None,
        parameter_generator=None
    )

    return exps
