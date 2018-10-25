import logging
import sys


def hill_climbing(s0, adjacencies, heuristic, goal_states):
    """ """

    # counts the number of loops (only for printing)
    s = s0
    plan = [s]
    while s not in goal_states:

        h_s = heuristic(s)
        improvement_found = False
        for succ in adjacencies[s]:
            h_succ = heuristic(succ)
            if h_succ < h_s:
                s = succ
                plan.append(s)
                improvement_found = True
                break

        if not improvement_found:
            # Error!
            successor_values = [(succ, heuristic(succ)) for succ in adjacencies[s]]
            print("", file=sys.stderr)
            print("h({})={}, but:".format(s, h_s), file=sys.stderr)
            print("\t" + "\n\t".join("h({})={}".format(succ, hsucc) for succ, hsucc in successor_values), file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError("Your model doesn't work")

    assert s in goal_states
    print("\nHill Climbing on the training instance succeeds with the following state path: ")
    print(", ".join(plan))


def create_pyperplan_hill_climbing_with_embedded_heuristic(heuristic):
    # This will import from pyperplan's search dir. A bit hacky. We might prefer to pass searchspace as a parameter
    # for something a bit cleaner
    from search import searchspace
    def pyperplan_hill_climbing(task):
        # assert False

        current = searchspace.make_root_node(task.initial_state)
        current_h = heuristic(current.state)

        while not task.goal_reached(current.state):

            improvement_found = False
            successors = task.get_successor_states(current.state)

            if not successors:
                logging.error("State without successors found: {}".format(current.state))
                return None

            for operator, succ in successors:
                h_succ = heuristic(succ)
                if h_succ < current_h:
                    current = searchspace.make_child_node(current, operator, succ)
                    improvement_found = True
                    break

            if not improvement_found:
                logging.error("Heuristic local minimum found on state {}".format(current.state))
                return None

        logging.info("Goal found")
        return current.extract_solution()

    return pyperplan_hill_climbing

