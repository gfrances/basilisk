from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR
from sltp.util.misc import update_dict



def experiments():
    domain = "domain.pddl"
    base = dict(
        domain_dir="miconic",
        domain=domain,
        benchmark_dir=PYPERPLAN_BENCHMARK_DIR
    )

    exps = dict()

    #
    exps["learn"] = update_dict(
        base,
        instances=["task01.pddl",
                   "task02.pddl",
                   "task03.pddl"],
        test_instances=[],
        test_domain=domain,
        num_tested_states=100000,
        num_states=100000, max_width=[-1],
        num_sampled_states=None,
        complete_only_wrt_optimal=True,
        max_concept_size=10, max_concept_grammar_iterations=3,
        concept_generator=None, parameter_generator=None,
        feature_namer=None )

    return exps
