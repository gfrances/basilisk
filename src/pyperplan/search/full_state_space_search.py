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
    """

    def __init__(self):
        self.states = dict()
        self.global_id = 0

    """
    Add a state to the state space information

    @param state: state from the SearchNode class
    @param parent: parent from the SearchNode class
    """
    def add_state(self, state, parent):
        # corner case for initial state
        if parent is None:
            self.states[state] = (self.global_id, [])
            self.global_id += 1
            return
        if state in list(self.states.keys()):
            state_id, parents_list = self.states[state]
            parents_list.append(self.get_id(parent.state))
            self.states[state] = state_id, parents_list
        else:
            self.states[state] = (self.global_id, [self.get_id(parent.state)])
            self.global_id += 1

    def get_id(self, state):
        state_id, _ = self.states[state]
        return state_id

    def _debug_print(self):
        print (self.states)

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
    closed = {planning_task.initial_state}
    while queue:
        iteration += 1
        logging.debug("breadth_first_search: Iteration %d, #unexplored=%d"
                      % (iteration, len(queue)))
        # get the next node to explore
        node = queue.popleft()


        # add state to our StateSpaceInfo
        state_space.add_state(node.state, node.parent)

        # exploring the node or if it is a goal node extracting the plan
        if planning_task.goal_reached(node.state):
            goals += 1
            logging.info("Goal found. Number of goal states found: %d" % goals)
        if iteration >= max_exp:
            logging.info("Maximum number of expansions reached. Exiting the search.")
            logging.info("Total number of goal states: %d" % goals)
            logging.info("%d Nodes expanded" % iteration)
            state_space._debug_print()
            return None
        for operator, successor_state in planning_task.get_successor_states(
                                                                   node.state):
            # duplicate detection
            if successor_state not in closed:
                queue.append(searchspace.make_child_node(node, operator,
                                                         successor_state))
                 # remember the successor state
                closed.add(successor_state)
    logging.info("No operators left.")
    logging.info("%d Nodes expanded" % iteration)
    logging.info("Total number of goal states: %d" % goals)
    state_space._debug_print()
    return None
