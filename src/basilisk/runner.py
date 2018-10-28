#!/usr/bin/env python3
import logging

import cplex

from .read_input import *


# Return the name of the weight variable given the feature
from .search import hill_climbing
from .utils import natural_sort


def get_weight_var(f):
    return 'w_' + str(f)


# Return the name of the binary aux variable given the transition
def get_y_var(s, t):
    return 'y_' + s + '_' + t


# Populate obj function and add all variables
def populate_obj_function(problem, num_features, max_weight, transitions, feature_complexity, goal_states):
    # Add variables that occur in the opt function
    for f in range(0, num_features):
        problem.variables.add(names=['xplus_' + str(f), 'xminus_' + str(f)],
                              obj=[feature_complexity[f], feature_complexity[f]],
                              types=[problem.variables.type.binary] * 2)

    # add weights
    for f in range(0, num_features):
        problem.variables.add(names=[get_weight_var(f)],
                              obj=[0],
                              lb=[-1 * max_weight], ub=[max_weight],
                              #                              types=[problem.variables.type.continuous])
                              types=[problem.variables.type.integer])

    # add binary to each transition
    for t in transitions:
        if t[0] in goal_states:
            continue

        problem.variables.add(names=[get_y_var(t[0], t[1])],
                              obj=[0],
                              types=[problem.variables.type.binary])

    problem.variables.add(names=["dummy"], obj=[0], lb=[1], ub=[1], types=[problem.variables.type.integer])


# Populate constraints (4a) in the draft
def populate_weight_constraints(problem, transitions, features, goal_states, max_weight):
    for transition_i, transition in enumerate(transitions, 1):

        coefficients = []
        names = []
        start_node = transition[0]
        final_node = transition[1]

        if start_node in goal_states:
            continue

        constraint_name = "c_" + start_node + "_" + final_node
        features_start = features[start_node]
        features_final = features[final_node]
        sum_delta_f = 0

        # Add w_f*([f]^s - [f]^s') to every feature
        for f, (start_f, final_f) in enumerate(zip(features_start, features_final)):
            w_name = get_weight_var(f)
            delta_f = start_f - final_f
            sum_delta_f += abs(delta_f)
            coefficients.append(delta_f)
            names.append(w_name)

        problem.indicator_constraints.add(
            indvar = get_y_var(start_node, final_node),
            complemented = 0,
            rhs = 1,
            sense = "G",
            lin_expr = cplex.SparsePair(ind=names, val=coefficients),
            name= constraint_name)


def populate_dead_end_constraints(problem, transitions, goal_states, dead_ends):
    for transition_i, transition in enumerate(transitions, 1):
        '''
        We are doing almost the same thing as the weight constraints here.
        Maybe we can save some work if we merge both functions?
        '''
        coefficients = []
        names = []
        start_node = transition[0]
        final_node = transition[1]

        if start_node not in dead_ends:
            continue

        constraint_name = "d_" + start_node + "_" + final_node
        features_start = features[start_node]
        features_final = features[final_node]
        sum_delta_f = 0

        for f, (start_f, final_f) in enumerate(zip(features_start, features_final)):
            w_name = get_weight_var(f)
            delta_f = start_f - final_f
            coefficients.append(delta_f)
            names.append(w_name)

        problem.linear_constraint.add(
            lin_expr = cplex.SparsePair(ind=names, val=coefficients),
            sense = "L",
            rhs = 0,
            name= constraint_name)


# Populate constraints (4b) in the draft
def populate_y_constraints(problem, adj_list, goal_states):
    for start in adj_list:
        if start in goal_states:
            continue
        constraint_name = "c_y_" + start
        problem.linear_constraints.add(names=[constraint_name])
        coefficients = []
        for final in adj_list[start]:
            var = get_y_var(start, final)
            coefficients.append((constraint_name, var, 1))
        problem.linear_constraints.set_coefficients(coefficients)
        problem.linear_constraints.set_rhs(constraint_name, 1)
        problem.linear_constraints.set_senses(constraint_name, "G")


# Populate constraints (7,8,9) in the draft
def populate_max_weight_constraints(problem, num_features, max_weight):
    coefficients = []
    for f in range(0, num_features):
        w_name = get_weight_var(f)

        # x_+ constraint
        xplus_name = 'c_xplus_' + str(f)
        problem.linear_constraints.add(names=[xplus_name])
        coefficients.append((xplus_name, w_name, -1))
        coefficients.append((xplus_name, 'xplus_' + str(f), max_weight))
        problem.linear_constraints.set_rhs(xplus_name, 0)
        problem.linear_constraints.set_senses(xplus_name, "G")

        # x_- constraint
        xminus_name = 'c_xminus_' + str(f)
        problem.linear_constraints.add(names=[xminus_name])
        coefficients.append((xminus_name, w_name, 1))
        coefficients.append((xminus_name, 'xminus_' + str(f), max_weight))
        problem.linear_constraints.set_rhs(xminus_name, 0)
        problem.linear_constraints.set_senses(xminus_name, "G")

        # w_f <= max_weight constraint
        c_max_weight_name = "c_max_w_" + w_name
        problem.linear_constraints.add(names=[c_max_weight_name])
        coefficients.append((c_max_weight_name, w_name, 1))
        coefficients.append((c_max_weight_name, "dummy", -1 * max_weight))
        problem.linear_constraints.set_rhs(c_max_weight_name, 0)
        problem.linear_constraints.set_senses(c_max_weight_name, "L")
    problem.linear_constraints.set_coefficients(coefficients)


def report(problem, transitions, feature_names, features_per_state, goal_states, adj_list, config):

    wvar_names = ((i, get_weight_var(i)) for i in range(0, len(feature_names)))
    feature_weights = ((i, wname, problem.solution.get_values(wname)) for i, wname in wvar_names)
    nonzero_features = [(i, val) for i, var, val in feature_weights if val != 0]
    print("Heuristic found making use of {} features:".format(len(nonzero_features)))
    print(("\t{}".format("\n\t".join("{} · {}".format(val, feature_names[i]) for i, val in nonzero_features))))

    def heuristic_function(s_):
        state_values = features_per_state[s_]
        h = 0
        for i, fweight in nonzero_features:
            h += fweight * state_values[i]
        return h

    sorted_state_ids = natural_sort(features_per_state.keys())
    with open(config.state_heuristic_filename, "w") as file:
        for s in sorted_state_ids:
            print("h({}) = {}".format(s, heuristic_function(s)), file=file)

    # print("Weight Variables:")
    # print("\n".join("{}: {}".format(var, val) for var, val in var_vals))

    # print("Y Variables:")
    # y_variables = [get_y_var(t[0], t[1]) for t in transitions if t[0] not in goal_states]
    # for i in y_variables:
    #     print(i, '=', str(int(round(problem.solution.get_values(i)))) + ', ', end=' ')
    # print()

    # print("X+- Variables:")
    # for f in range(0, len(feature_names)):
    #     print(f, str(
    #         (problem.solution.get_values('xplus_' + str(f)), problem.solution.get_values('xminus_' + str(f)))) + ', ',
    #           end=" ")

    # Run hill-climbing and make sure we find a goal
    hill_climbing("s0", adj_list, heuristic_function, goal_states)
    return nonzero_features


def run(config, data, rng):
    max_weight = config.lp_max_weight

    transitions, adj_list = read_transition_file(config.transitions_filename)
    features_per_state, num_features = read_features_file(config.feature_matrix_filename)
    goal_states = read_goal_states(config.goal_states_filename)
    feature_complexity, feature_names = read_complexity_file(config.feature_info_filename)
    dead_ends = read_unsolvable_states(None) # TODO add file name

    logging.info("Read {} transitions, {} features, {} goal states".
                 format(len(transitions), len(goal_states), num_features))

    logging.info("Populating model")
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)

    logging.info("Populating objective function")
    populate_obj_function(problem, num_features, max_weight, transitions, feature_complexity, goal_states)
    logging.info("Populating weight constraints")
    populate_weight_constraints(problem, transitions, features_per_state, goal_states, max_weight)
    logging.info("Populating dead-end constraints")
    populate_dead_end_constraints(problem, transitions, goal_states, dead_ends)
    logging.info("Populating y-constraints")
    populate_y_constraints(problem, adj_list, goal_states)
    logging.info("Populating M_w-constraints")
    populate_max_weight_constraints(problem, num_features, max_weight)

    logging.info("Writing file...")
    problem.write(config.lp_filename)

    logging.info("Solving MIP...")
    problem.solve()
    if problem.solution.is_primal_feasible() and problem.solution.is_dual_feasible():
        print ("Optimal solution found.")
        print ("Solution value  = {}".format(problem.solution.get_objective_value()))
        heuristic = report(problem, transitions, feature_names, features_per_state, goal_states, adj_list, config)

        # Return those values that we want to be persisted between different steps
        return dict(
            learned_heuristic=heuristic,
        )
    elif problem.solution.is_primal_feasible():
        print ("Problem is unbounded")
    elif problem.solution.is_dual_feasible():
        logging.info("Problem is unsolvable")
    else:
        logging.info("Problem was not solved. Unknown reason.")

    # How should we proceed if it has no solution?
    sys.exit(1)
