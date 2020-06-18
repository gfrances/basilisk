from basilisk import PYPERPLAN_BENCHMARK_DIR
from sltp.util.misc import update_dict


def experiments():
    domain = "domain.pddl"
    base = dict(
        domain_dir="miconic",
        domain=domain,
        benchmark_dir=PYPERPLAN_BENCHMARK_DIR
    )

    exps = dict()

    exps["small"] = update_dict(
        base,
        instances=["task01.pddl",
                   "task02.pddl",
                   "task03.pddl"],
        test_instances=[
            "task10.pddl",
            "task15.pddl",
            "task20.pddl",
            "task25.pddl",
            "task30.pddl",
        ],
        test_domain=domain,
        num_states=100000,
        num_sampled_states=None,
        complete_only_wrt_optimal=True,
        max_concept_size=50,
        concept_generation_timeout=1800,  # in seconds
        concept_generator=None,
        parameter_generator=None,
        feature_namer=None)

    return exps
