import logging
import sys


def hill_climbing(s0, adjacencies, heuristic, goal_states):
    """ """

    # counts the number of loops (only for printing)
    s = s0
    plan = []

    while s not in goal_states:

        h_s = heuristic(s)
        plan.append((s, h_s))
        improvement_found = False
        for succ in adjacencies[s]:
            h_succ = heuristic(succ)
            if h_succ < h_s:
                s = succ
                # plan.append(s)
                improvement_found = True
                break

        if not improvement_found:
            # Error!
            successor_values = [(succ, heuristic(succ)) for succ in adjacencies[s]]
            logging.error("Error found while validating learnt heuristic: hill-climbings gets stuck in local minimum")
            logging.error("Path so far was: {}".format(plan))
            print("", file=sys.stderr)
            print("h({})={}, but:".format(s, h_s), file=sys.stderr)
            print("\t" + "\n\t".join("h({})={}".format(succ, hsucc) for succ, hsucc in successor_values), file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError("Your model doesn't work")

    assert s in goal_states
    print("\nHill Climbing on the training instance succeeds with the following state path: ")
    print(", ".join("{} ({})".format(s, h) for s, h in plan))

def parse_atom(atom):
    atom = atom.replace('(', '')
    atom = atom.replace(')', '')
    split = atom.split()
    action = split[0]
    new_atom = action+"(" + (",".join(split[1:])) + ")"
    return new_atom

def create_pyperplan_hill_climbing_with_embedded_heuristic(heuristic):
    # This will import from pyperplan's search dir. A bit hacky. We might prefer to pass searchspace as a parameter
    # for something a bit cleaner
    from search import searchspace
    def pyperplan_hill_climbing(task):
        # assert False

        current = searchspace.make_root_node(task.initial_state)
        current_h = heuristic(current.state)

        iterations = 0

        # Keep path and successor states of detected flaw
        pyperplan_hill_climbing.path = []
        pyperplan_hill_climbing.dead_end_succs = []

        while not task.goal_reached(current.state):
            iterations += 1
            if iterations % 1000 == 0:
                logging.debug("Number of visited states in hill-climbing: {}".format(iterations))

            improvement_found = False
            successors = task.get_successor_states(current.state)

            pyperplan_hill_climbing.path.append([parse_atom(a) for a in current.state])

            if not successors:
                logging.error("State without successors found: {}".format(current.state))
                return None

            # This implements steepest-ascent hill climbing.
            # If we want to test in problems with very large branching factors, we might want to jump to
            # an heuristic-improving child as soon as we find one.
            succesor_h_values = [(heuristic(succ), succ, op) for op, succ in successors]
            h_succ, succ, op = min(succesor_h_values, key=lambda x: x[0])

            if h_succ < current_h:
                current = searchspace.make_child_node(current, op, succ)
                current_h = h_succ
            else:
                logging.error("Heuristic local minimum of {} found on state {}".format(current_h, current.state))
                logging.error("Children nodes:")
                print("\t" + "\n\t".join("h: {}, s: {}".format(h, s) for h, s, _ in succesor_h_values))
                for _, succ in successors:
                    atoms = [parse_atom(atom) for atom in succ]
                    pyperplan_hill_climbing.dead_end_succs.append(atoms)
                return None

        logging.info("Goal found")
        return current.extract_solution()

    return pyperplan_hill_climbing
