
import os

from basilisk import BENCHMARK_DIR
from sltp.driver import Experiment, generate_pipeline_from_list, \
    TransitionSamplingStep, PlannerStep, CPPFeatureGenerationStep

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# The steps necessary to generate the data to train the neural network;
# h* and feature denotation matrix
NN_TRAINING_DATA_PIPELINE = [
    PlannerStep,
    TransitionSamplingStep,
    CPPFeatureGenerationStep,
]


def merge_parameters_with_defaults(domain_dir, domain, **kwargs):
    if "instances" not in kwargs:
        raise RuntimeError("Please specify domain and instance when generating an experiment")

    instances = kwargs["instances"] if isinstance(kwargs["instances"], (list, tuple)) else [kwargs["instances"]]
    benchmark_dir = kwargs.get("benchmark_dir", BENCHMARK_DIR)
    kwargs["domain"] = os.path.join(benchmark_dir, domain_dir, domain)
    kwargs["test_domain"] = os.path.join(benchmark_dir, domain_dir, kwargs["test_domain"])
    kwargs["instances"] = [os.path.join(benchmark_dir, domain_dir, i) for i in instances]
    kwargs["test_instances"] = [os.path.join(benchmark_dir, domain_dir, i) for i in kwargs["test_instances"]]

    defaults = dict(
        # Location of the FS planner, used to do the state space sampling
        planner_location=os.getenv("FS_PATH", os.path.expanduser("~/projects/code/fs")),

        # The directory where the experiment outputs will be left
        workspace=os.path.join(BASEDIR, 'workspace'),

        # Type of sampling procedure. Only breadth-first search implemented ATM
        driver="bfs",

        # Whether we are happy to obtain completeness guarantees only with respect to those states
        # that lie in some arbitrary (one) optimal path. Default: False
        complete_only_wrt_optimal=False,

        # Type of sampling. Accepted options are:
        # - "all" (default): Use all expanded states
        # - "random": Use only a random sample of the expanded states, of size given by the option "num_sampled_states"
        # - "optimal": Use those expanded states on some optimal path (only one arbitrary optimal path)
        sampling="all",

        # Number of states to be expanded in the sampling procedure
        num_states=50,

        # Number randomly sampled states among the set of expanded states. The default of None means
        # all expanded states will be used
        num_sampled_states=None,

        # Max. size of the generated concepts (mandatory)
        max_concept_size=10,

        # Provide a special, handcrafted method to generate concepts, if desired.
        # This will override the standard concept generation procedure (default: None)
        concept_generator=None,

        # Or, alternatively, provide directly the features instead of the concepts (default: None)
        feature_generator=None,

        # A list of features the user wants to be in the abstraction (possibly along others).
        # Default is [], i.e. just run the generation process without enforcing anything
        enforce_features=[],

        # Max. allowed complexity for distance and conditional features (default: 0)
        distance_feature_max_complexity=0,
        cond_feature_max_complexity=0,

        # Method to generate domain parameters (goal or otherwise). If None, goal predicates will
        # be used (default: None)
        parameter_generator=None,

        # Optionally, use a method that gives handcrafted names to the features
        # (default: None, which will use their string representation)
        feature_namer=default_feature_namer,

        # Set a random seed for reproducibility (default: 1)
        random_seed=1,

        domain_dir=domain_dir,

        # Whether to clean the (sub-) workspaces used to compute incremental abstractions after finishing.
        clean_workspace=True,

        # Reduce output to a minimum
        quiet=False,

        # Some debugging help to print the denotations of all features over all states (expensive!)
        print_all_denotations=False,

        # Max. time in seconds for the concept generation phase. A value of -1 implies no timeout.
        concept_generation_timeout=-1,


        # We want to keep this to 0, as we don't want to sample states from testing instances
        num_tested_states=0,

        # Keep this to -1, as we don't want any kind of width-based exploration
        max_width=[-1],

        # Ignore this attributes, they are necessary for SLTP to run without complaining
        optimal_selection_strategy="arbitrary",
        create_goal_features_automatically=False,
        goal_selector=None,
    )

    parameters = {**defaults, **kwargs}  # Copy defaults, overwrite with user-specified parameters
    return parameters
    
    
def generate_experiment(domain_dir, domain, **kwargs):
    """ """
    parameters = merge_parameters_with_defaults(domain_dir, domain, **kwargs)
    steps, parameters = generate_pipeline_from_list(elements=NN_TRAINING_DATA_PIPELINE, **parameters)
    exp = Experiment(steps, parameters)
    return exp


def default_feature_namer(s):
    return str(s)
