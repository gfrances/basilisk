import os
import sys

from . import EXPDATA_DIR, PYPERPLAN_DIR
from sltp.driver import Step, InvalidConfigParameter, check_int_parameter
from sltp.util.command import execute
from sltp.util.naming import compute_instance_tag, compute_experiment_tag, compute_sample_filename


def compute_sample_filenames(experiment_dir, instances, **_):
    return [os.path.join(experiment_dir, compute_sample_filename(i, "0")) for i in instances]


def _run_pyperplan(config, data, rng):
    # Run the planner on all the instances

    # config.num_states

    for i, o in zip(config.instances, config.sample_files):
        # params = '-i {} --domain {} --driver {} --disable-static-analysis --options="max_expansions={},width.max={}"'\
        #     .format(i, config.domain, config.driver, config.num_states, w)
        max_expansions = "full"  # config.num_states
        params = '-s {} {} {}'.format(max_expansions, config.domain, i)
        execute(command=[sys.executable, "pyperplan.py"] + params.split(' '), stdout=o, cwd=PYPERPLAN_DIR)
    return dict()


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
        config["experiment_dir"] = os.path.join(EXPDATA_DIR, config["experiment_tag"])
        config["sample_files"] = compute_sample_filenames(**config)

        # TODO This should prob be somewhere else:
        os.makedirs(config["experiment_dir"], exist_ok=True)

        return config

    def description(self):
        return "Sampling of the state space"

    def get_step_runner(self):
        return _run_pyperplan
