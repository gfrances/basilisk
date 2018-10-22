#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

'''
Implements a complete search until a given limit of expansions.
'''

from collections import deque
import logging


from . import searchspace

class StateSpaceInfo:
    """
    Auxiliary class just to keep track of the transitions and
    expanded states.
    Similar to the searchspace.py classes, but used only
    for our purposes.

    NOTE: the data structure used during the search is different from the one
    printed and used later, we might be able to make it the same

    """

    def __init__(self):
        self.states = dict()
        self.global_id = 0

    """
    Add a state to the state space information

    @param state: state from the SearchNode class
    @param parent: parent from the SearchNode class
    """
    def add_state(self, state, parent, is_goal):
        # corner case for initial state
        if parent is None:
            self.states[state] = (self.global_id, [], is_goal)
            self.global_id += 1
            return
        if state in list(self.states.keys()):
            state_id, parents_list, is_goal = self.states[state]
            parents_list.append(self.get_id(parent.state))
            self.states[state] = (state_id, parents_list, is_goal)
        else:
            self.states[state] = (self.global_id, [self.get_id(parent.state)], is_goal)
            self.global_id += 1

    def get_id(self, state):
        state_id, _, _ = self.states[state]
        return state_id

    def _debug_print(self):
        for i in self.states:
            print (i, self.states[i])

    def parse_atom(self, atom):
        atom = atom.replace('(', '')
        atom = atom.replace(')', '')
        split = atom.split()
        action = split[0]
        new_atom = action+"(" + (",".join(split[1:])) + ")"
        return new_atom


    def convert_to_json(self):
        output = []
        for state in self.states:
            state_id, parents, is_goal = self.states[state]
            atoms = [self.parse_atom(atom) for atom in state]
            for parent in parents:
                new_entry = dict()
                new_entry["id"] = state_id
                new_entry["parent"] = parent
                new_entry["goal"] = is_goal
                new_entry["atoms"] = atoms
                output.append(new_entry)
        return output


def full_state_space_search(planning_task, max_exp=10000):

    # create obj to track state space
    state_space = StateSpaceInfo()

    # counts the number of loops (only for printing) and goal states found
    iteration = 0
    goals = 0
    # fifo-queue storing the nodes which are next to explore
    queue = deque()
    queue.append(searchspace.make_root_node(planning_task.initial_state))
    # set storing the explored nodes, used for duplicate detection

    closed = set()
    node_id = 0
    while queue:
        iteration += 1
        logging.debug("breadth_first_search: Iteration %d, #unexplored=%d"
                      % (iteration, len(queue)))
        # get the next node to explore
        node = queue.popleft()
        is_goal = planning_task.goal_reached(node.state)

        # add state to our StateSpaceInfo or update its parent
        state_space.add_state(node.state, node.parent, is_goal)

        # we manage the closed list here to allow the parents update
        if node.state in closed:
            # if it has already been expanded, skip
            continue
        else:
            closed.add(node.state)

        node_id += 1

        # exploring the node or if it is a goal node extracting the plan
        if is_goal:
            goals += 1
            logging.info("Goal found. Number of goal states found: %d" % goals)
        if node_id >= max_exp:
            logging.info("Maximum number of expansions reached. Exiting the search.")
            logging.info("Total number of goal states: %d" % goals)
            logging.info("%d Nodes expanded" % node_id)
            return state_space.convert_to_json()
        for operator, successor_state in planning_task.get_successor_states(node.state):
            queue.append(searchspace.make_child_node(node, operator,
                                                     successor_state))
    logging.info("No operators left.")
    logging.info("%d Nodes expanded" % node_id)
    logging.info("Total number of goal states: %d" % goals)

    return state_space.convert_to_json()
