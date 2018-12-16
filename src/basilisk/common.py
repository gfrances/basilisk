
from sltp.features import DLModelCache
from sltp.sampling import TransitionSample


def is_alive(sid, sample: TransitionSample):
    """ A state is alive if it is solvable, reachable, and not a goal state
    Given that all our samples are obtained by a forward-search exploration, any state in the sample
    will be implicitly reachable, so we just need to check the remaining two conditions
    """
    return sid not in sample.unsolvable and sid not in sample.goals


def has_improving_successor(s, h_s, sample: TransitionSample, model_cache: DLModelCache, heuristic):
    """ Check that the given heuristic has an improving successor *on the given state*
    h_s is already given for performance reasons.
     """
    for sprime in sample.transitions[s]:
        mprime = model_cache.get_feature_model(sprime)
        hprime = heuristic.value(mprime)
        if hprime < h_s:
            return True
    return False
