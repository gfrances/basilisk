import json
import os
import sys

from .tester import compute_static_atoms
from sltp.features import parse_pddl
from sltp.returncodes import ExitCode

from . import EXPDATA_DIR, PYPERPLAN_DIR
from sltp.driver import Step, InvalidConfigParameter, check_int_parameter
from sltp.util.command import execute, read_file
from sltp.util.naming import compute_instance_tag, compute_experiment_tag, compute_sample_filename, \
    compute_info_filename


def compute_sample_filenames(experiment_dir, instances, **_):
    return [os.path.join(experiment_dir, compute_sample_filename(i, "0")) for i in instances]


def _run_pyperplan(config, data, rng):
    # Run the planner on all the instances

    # config.num_states
    print ("Using experiment directory %s" % config.experiment_dir)
    for i, o in zip(config.instances, config.sample_files):
        # params = '-i {} --domain {} --driver {} --disable-static-analysis --options="max_expansions={},width.max={}"'\
        #     .format(i, config.domain, config.driver, config.num_states, w)
        params = '-s full --state-space-output {} --max-nodes {} {} {}'.format(o, config.num_states, config.domain, i)
        execute(command=[sys.executable, "pyperplan.py"] + params.split(' '), cwd=PYPERPLAN_DIR)

        problem, _, _ = parse_pddl(config.domain, i)
        static_atoms, static_predicates = compute_static_atoms(problem)

        all_states = []
        # Extend pyperplan output with the necessary static predicates:
        # read json output
        for line in read_file(o):
            state = json.loads(line)
            assert all(len(static) > 0 for static in static_atoms)
            state["atoms"] += ["{}({})".format(static[0], ','.join(static[1:])) for static in static_atoms]
            all_states.append(state)

        with open(o, "w") as f:
            for s in all_states:
                print(json.dumps(s), file=f)

    return ExitCode.Success, dict()


class PyperplanStep(Step):
    """ Run some planner on certain instance(s) to get the sample of transitions """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_required_attributes(self):
        return ["instances", "domain", "num_states", "planner_location", "driver"]

    def get_required_data(self):
        return []

    def process_config(self, config):
        if any(not os.path.isfile(i) for i in config["instances"]):
            raise InvalidConfigParameter('"instances" must be the path to existing instance files')
        if not os.path.isfile(config["domain"]):
            raise InvalidConfigParameter('"domain" must be the path to an existing domain file')
        check_int_parameter(config, "num_states", positive=True)

        config["instance_tag"] = compute_instance_tag(**config)
        config["experiment_tag"] = compute_experiment_tag(**config)
        config["experiment_dir"] = os.path.join(EXPDATA_DIR, config["experiment_tag"][:64])
        config["sample_files"] = compute_sample_filenames(**config)

        # TODO This should prob be somewhere else:
        os.makedirs(config["experiment_dir"], exist_ok=True)

        return config

    def description(self):
        return "Sampling of the state space with Pyperplan"

    def get_step_runner(self):
        return _run_pyperplan


class HeuristicWeightsLPComputation(Step):
    """  """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_required_attributes(self):
        return ["experiment_dir", "lp_max_weight", "transitions_filename", "feature_matrix_filename",
                "goal_states_filename", "feature_info_filename", "unsolvable_states_filename"]

    def process_config(self, config):
        config["lp_filename"] = compute_info_filename(config, "problem.lp")
        config["state_heuristic_filename"] = compute_info_filename(config, "state-heuristic-values.txt")

        return config

    def get_required_data(self):
        return ["features"]

    def description(self):
        return "Computation of the weights of a desceding heuristic"

    def get_step_runner(self):
        from . import runner
        return runner.run


class HeuristicTestingComputation(Step):
    """  """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_required_attributes(self):
        return ["experiment_dir", "test_instances", "test_domain"]

    def process_config(self, config):
        if not os.path.isfile(config["test_domain"]):
            raise InvalidConfigParameter('"test_domain" must be the path to an existing domain file')
        if any(not os.path.isfile(i) for i in config["test_instances"]):
            raise InvalidConfigParameter('"test_instances" must point to existing files')

        return config

    def get_required_data(self):
        return ["learned_heuristic"]

    def description(self):
        return "Testing of the heuristic in unseen instances"

    def get_step_runner(self):
        from . import tester
        return tester.run
