import logging
import os
import shutil
from collections import defaultdict

import numpy as np
from basilisk.common import is_alive, has_improving_successor
from basilisk.runner import ConceptBasedPotentialHeuristic
from basilisk.steps import PyperplanStep, HeuristicWeightsLPComputation, HeuristicTestingComputation
from sltp.driver import Experiment, generate_pipeline_from_list, check_int_parameter, InvalidConfigParameter, load, \
    TransitionSamplingStep, run_and_check_output, SubprocessStepRunner, Bunch, save, ConceptGenerationStep, \
    FeatureMatrixGenerationStep
from sltp.features import create_model_cache_from_samples, parse_all_instances, report_use_goal_denotation, \
    compute_instance_information
from sltp.returncodes import ExitCode
from sltp.sampling import log_sampled_states
from sltp.util.misc import update_dict
from tarski.dl import compute_dl_vocabulary


def initial_sample_selection(sample, config, rng):
    # Get the initial sample of M states plus one goal state per instance, if any goal is available,
    # and all instance roots
    # all_state_ids_shuffled = list(sample.states.keys())
    # rng.shuffle(all_state_ids_shuffled)
    expanded_state_ids_shuffled = list(sample.expanded)
    rng.shuffle(expanded_state_ids_shuffled)
    initial_size = min(len(expanded_state_ids_shuffled), config.initial_sample_size-len(sample.roots))
    working_sample_idxs = set(expanded_state_ids_shuffled[:initial_size])
    working_sample_idxs.update(sample.get_one_goal_per_instance())
    working_sample_idxs.update(sample.roots)
    return working_sample_idxs, expanded_state_ids_shuffled


def full_learning(config, sample, rng):
    working_sample_idxs, expanded_state_ids_shuffled = initial_sample_selection(sample, config, rng)
    working_sample = sample.resample(working_sample_idxs)

    use_goal_denotation = report_use_goal_denotation(config.parameter_generator)

    parsed_problems = parse_all_instances(config.domain, config.instances)
    infos = [compute_instance_information(problem, use_goal_denotation) for problem, _, _ in parsed_problems]

    # We assume all problems languages are the same and simply pick the first one
    language = parsed_problems[0][0].language
    vocabulary = compute_dl_vocabulary(language)

    # Note that the validator will validate wrt the full sample
    _, model_cache = create_model_cache_from_samples(vocabulary, sample, config.domain, config.parameter_generator, infos)
    validator = KnowledgeValidator(model_cache, sample, expanded_state_ids_shuffled)

    k, k_max, k_step = config.initial_concept_bound, config.max_concept_bound, config.concept_bound_step
    assert k <= k_max
    while True:
        print("Working sample idxs: {}".format(sorted(working_sample_idxs)))
        print("Working sample: {}".format(working_sample.info()))
        print("States in Working sample: {}".format(sorted(working_sample.remapping.keys())))
        res, k, abstraction = try_to_compute_heuristic_in_range(config, working_sample, k, k_max, k_step)
        if res == ExitCode.NoAbstractionUnderComplexityBound:
            logging.error("No abstraction possible for given sample set under max. complexity {}".format(k_max))
            return res, dict()

        # Otherwise, we learnt an abstraction with max. complexity k. Let's refine the working sample set with it!
        logging.info("Abstraction with k={} found for sample set of size {} ".format(k, working_sample.num_states()))

        # Look for flaws in the full sample set
        flaws = validator.find_flaws(abstraction, config.batch_refinement_size)
        if not flaws:
            break
        logging.info("{} flaws found in the computed abstraction: {}".format(len(flaws), sorted(flaws)))

        # Augment the sample set with the flaws found for the current abstraction
        working_sample_idxs.update(flaws)
        working_sample = sample.resample(working_sample_idxs)  # This will "close" the sample set with children states

    logging.info("The computed abstraction is sound & complete wrt all of the training instances")
    return ExitCode.Success, abstraction


def try_to_compute_heuristic_in_range(config, sample, k_0, k_max, k_step):
    k, abstraction = k_0, None  # To avoid referenced-before-assigned warnings
    for k in range(k_0, k_max + 1, k_step):
        exitcode, abstraction = try_to_compute_heuristic(config, sample, k)
        if exitcode == ExitCode.Success:
            return exitcode, k, abstraction
        else:
            logging.info("No abstraction possible with max. concept complexity {}".format(k))
    return ExitCode.NoAbstractionUnderComplexityBound, k, abstraction


def setup_workspace(config, sample, k):
    """ Create the workspace and update the config for the incremental search with the given parameters """
    # Create a subdir, e.g. inc_k5_s20, to place all the data for this learning attempt
    subdir = os.path.join(config.experiment_dir, "inc_k{}_s{}".format(k, sample.num_states()))
    os.makedirs(subdir, exist_ok=True)
    save(subdir, dict(sample=sample))
    log_sampled_states(sample, os.path.join(subdir, "sample.txt"))
    logging.info("Search for abstraction: |S|={}, k=|{}|".format(sample.num_states(), k))
    logging.info("Data stored in {}".format(subdir))
    subconfig = update_dict(config.to_dict(), max_concept_size=k, experiment_dir=subdir)
    return subconfig


def teardown_workspace(experiment_dir, clean_workspace, **kwargs):
    """ Clean up the given workspace """
    data = load(experiment_dir, ["learned_heuristic"])
    # if clean_workspace:
    #     print(experiment_dir)
    #     shutil.rmtree(experiment_dir)  # TODO activate
    return data


def try_to_compute_heuristic(config, sample, k):
    """ Try to learn a generalized potential heursitic for the given sample,
    with max. concept complexity given by k
    """
    subconfig = setup_workspace(config, sample, k)

    steps, subconfig = generate_pipeline_from_list([
        ConceptGenerationStep, FeatureMatrixGenerationStep,
        HeuristicWeightsLPComputation],
        **subconfig)

    for step in steps:
        exitcode = run_and_check_output(step, SubprocessStepRunner, raise_on_error=False)
        if exitcode != ExitCode.Success:
            return exitcode, None

    # All steps successfully executed, ergo we found an abstraction.
    data = teardown_workspace(**subconfig)
    return ExitCode.Success, data


class KnowledgeValidator:
    def __init__(self, model_cache, sample, state_ids):
        """ """
        self.model_cache = model_cache
        self.sample = sample
        self.state_ids = state_ids

    def find_flaws(self, abstraction, max_flaws):
        """ We iterate over all the states in the *full* training sample, and check whether the learnt heuristic
        is descending and dead-end-avoiding at each state s in the sample. If it is not, then we add s to the set
        of flaws.
         """
        sample = self.sample
        flaws = set()
        heuristic = abstraction["learned_heuristic"]
        assert isinstance(heuristic, ConceptBasedPotentialHeuristic)

        logging.info("Looking for flaws in heuristic:\n{}".format(heuristic))

        assert all(s in sample.expanded for s in self.state_ids)  # state_ids assumed to contain only expanded states.
        for sid in self.state_ids:
            model = self.model_cache.get_feature_model(sid)
            h_s = heuristic.value(model)

            # Check heuristic is descending in the current state
            if is_alive(sid, sample) and not has_improving_successor(sid, h_s, self.sample, self.model_cache, heuristic):
                flaws.add(sid)

            # TODO - CHECK DEAD-END-AVOIDING

            if len(flaws) >= max_flaws:
                break

        return flaws


class IncrementalLearner:
    """ Generate concepts, compute feature set, compute A_F, all in one """
    def __init__(self, **kwargs):
        self.config = Bunch(self.process_config(kwargs))

    def get_required_attributes(self):
        return ["domain", "experiment_dir", "initial_concept_bound", "max_concept_bound", "concept_bound_step", "initial_sample_size"]

    def get_required_data(self):
        return ["sample"]

    def process_config(self, config):
        check_int_parameter(config, "initial_concept_bound")
        check_int_parameter(config, "concept_bound_step")
        check_int_parameter(config, "max_concept_bound")

        if config["initial_concept_bound"] > config["max_concept_bound"]:
            raise InvalidConfigParameter("initial_config_bound ({}) must be <= than max_concept_bound ({})".
                                         format(config["initial_concept_bound"], config["max_concept_bound"]))
        return config

    def run(self):
        data = load(self.config.experiment_dir, ["sample"])  # Load the initial sample
        rng = np.random.RandomState(self.config.random_seed)
        exitcode, knowledge = full_learning(self.config, data["sample"], rng)
        save(self.config.experiment_dir, knowledge)
        return exitcode


class IncrementalExperiment(Experiment):
    """ This class handcodes the sequence of steps to be run, as the logic behind incremental running is
        more complex.
    """
    def __init__(self, steps, parameters):
        super().__init__([], parameters)  # Simply ignore the steps
        self.parameters = parameters

    def run(self, args=None):
        # We ignore whatever specified in the command line and run the incremental algorithm
        self.hello(args)

        # cannot run HC in the incremental approach, because we will sample the transition system
        self.parameters['validate_learnt_heuristic'] = False
        self.parameters['compute_unsolvable_states'] = False

        # 1. Extract and resample the whole training set
        initial_steps, config = generate_pipeline_from_list([PyperplanStep, TransitionSamplingStep], **self.parameters)
        exitcode = run_and_check_output(initial_steps[0], SubprocessStepRunner)
        exitcode = run_and_check_output(initial_steps[1], SubprocessStepRunner)

        # 2. Perform the incremental acquisition of knowledge
        learner = IncrementalLearner(**config)
        exitcode = learner.run()

        # exitcode = ExitCode.Success
        # 3. If the learning was successful, test it on some testing instances
        if exitcode == ExitCode.Success:
            steps, config = generate_pipeline_from_list([HeuristicTestingComputation], **config)
            exitcode = run_and_check_output(steps[0], SubprocessStepRunner)

        return exitcode
