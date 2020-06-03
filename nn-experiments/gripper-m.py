from basilisk import PYPERPLAN_BENCHMARK_DIR, BENCHMARK_DIR
from sltp.util.misc import update_dict


def experiments():
    domain = "domain.pddl"
    base = dict(
        domain_dir="gripper-m",
        domain=domain,
        benchmark_dir=BENCHMARK_DIR
    )

    exps = dict()

    #
    exps["learn"] = update_dict(
        base,
        instances=['testX.pddl',
                   'testY.pddl',
                   'testZ.pddl'],
        test_instances=[],
        test_domain=domain,
        num_tested_states=100000,
        num_states=100000, max_width=[-1],
        num_sampled_states=None,
        complete_only_wrt_optimal=True,
        max_concept_size=666,
        concept_generation_timeout=60,  # in seconds
        concept_generator=None, parameter_generator=add_domain_parameters,
        feature_namer=None
    )

    return exps


def add_domain_parameters(language):
    return [language.constant("roomb", "object")]


def feature_namer(feature):
    s = str(feature)
    return {
        "card[Exists(at,Not({roomb}))]": "nballs-A",
        "card[Exists(at,{roomb})]": "nballs-B",
        "card[Exists(carry,<universe>)]": "ncarried",
        "bool[And(at-robby,{roomb})]": "robot-at-B",
        "card[Exists(at,Not(at-robby))]": "nballs-in-rooms-with-no-robot",
        "card[free]": "nfree-grippers",
        "bool[Exists(at-robby,{roomb})]": "robby-is-at-B",
        "card[Exists(at,Exists(Inverse(at-robby),<universe>))]": "nballs-in-room-with-some-robot",
        "card[And(Exists(gripper,Exists(at-robby,{roomb})),free)]": "nfree-grippers-at-B",
        "card[Exists(at-robby,{roomb})]": "nrobots-at-B",
        "card[Exists(gripper,Exists(at-robby,{roomb}))]": "ngrippers-at-B",
        "card[Exists(carry,Exists(gripper,Exists(at-robby,{roomb})))]": "nballs-carried-in-B",
        "card[Exists(at,And(Forall(Inverse(at-robby),<empty>), Not({roomb})))]": "nballs-in-some-room-notB-without-any-robot",
        "bool[And(Exists(Inverse(at),<universe>), And({roomb}, Not(at-robby)))]": "some-ball-in-B-but-robbot-not-in-B",
    }.get(s, s)
