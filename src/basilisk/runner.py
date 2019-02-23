#!/usr/bin/env python3
import logging

import cplex
import math
import sys
from sltp.errors import CriticalPipelineError
from sltp.returncodes import ExitCode
from tarski.dl import EmpiricalBinaryConcept, ConceptCardinalityFeature

from .read_input import *
from .search import hill_climbing
from .utils import natural_sort


def get_weight_var(f):
    """ Return the name of the weight variable given the feature """
    return 'w_' + str(f)


def get_y_var(s, t):
    """ Return the name of the binary aux variable given the transition """
    return 'y_' + s + '_' + t


# Populate obj function and add all variables
def populate_obj_function_min_complexity(problem, num_features, max_weight, transitions, feature_complexity, goal_states):
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
                              types=[problem.variables.type.continuous])
                              #types=[problem.variables.type.integer])

    # add binary to each transition
    for t in transitions:
        if t[0] in goal_states:
            continue

        problem.variables.add(names=[get_y_var(t[0], t[1])],
                              obj=[0],
                              types=[problem.variables.type.binary])


# Populate obj function
def populate_obj_function_max_nonselected_complexity(problem, num_features, max_weight, transitions, feature_complexity, goal_states):
    # Add variables that occur in the opt function
    for f in range(0, num_features):
        problem.variables.add(names=['z_w_' + str(f)],
                              obj=[feature_complexity[f]],
                              types=[problem.variables.type.binary])


    # add weights
    for f in range(0, num_features):
        problem.variables.add(names=[get_weight_var(f)],
                              obj=[0],
                              lb=[-1 * cplex.infinity], ub=[cplex.infinity],
                              types=[problem.variables.type.continuous])

    # add binary to each transition
    for t in transitions:
        if t[0] in goal_states:
            continue

        problem.variables.add(names=[get_y_var(t[0], t[1])],
                              obj=[0],
                              types=[problem.variables.type.binary])

    problem.variables.add(names=["dummy"], obj=[0], lb=[1], ub=[1], types=[problem.variables.type.integer])

# Populate obj function
def populate_obj_function_min_weighted_complexity(problem, num_features, max_weight, transitions, feature_complexity, goal_states):
    # Add variables that occur in the opt function
    for f in range(0, num_features):
        problem.variables.add(names=['abs_w_' + str(f)],
                              obj=[feature_complexity[f]],
                              lb=[0],
                              ub=[1 * cplex.infinity],
                              types=[problem.variables.type.continuous])

    # add weights
    for f in range(0, num_features):
        problem.variables.add(names=[get_weight_var(f)],
                              lb=[-1 * cplex.infinity], ub=[cplex.infinity],
                              obj=[0],
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
def populate_weight_constraints(problem, transitions, features, goal_states, unsolvable_states, max_weight):
    for start_node, final_node in transitions:

        coefficients = []
        names = []

        if start_node in goal_states or start_node in unsolvable_states:
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

        # Computes M
        big_m = 1 + max_weight*sum_delta_f
        # Add term -M*y_s_s'
        names.append(get_y_var(start_node, final_node))
        coefficients.append(-1*big_m)

        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=names, val=coefficients)],
            senses=["G"],
            rhs=[-1*big_m+1],
            names=[constraint_name])


def populate_dead_end_constraints(problem, transitions, features, goal_states, unsolvable_states):
    for start_node, final_node in transitions:
        '''
        We are doing almost the same thing as the weight constraints here.
        Maybe we can save some work if we merge both functions?
        '''
        coefficients = []
        names = []

        if start_node in unsolvable_states or final_node not in unsolvable_states:
            continue

        constraint_name = "d_" + start_node + "_" + final_node
        features_start = features[start_node]
        features_final = features[final_node]

        for f, (start_f, final_f) in enumerate(zip(features_start, features_final)):
            w_name = get_weight_var(f)
            delta_f = start_f - final_f
            coefficients.append(delta_f)
            names.append(w_name)

        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=names, val=coefficients)],
            senses=["L"],
            rhs=[0],
            names=[constraint_name])


# Populate constraints (4b) in the draft
def populate_y_constraints(problem, adj_list, goal_states, unsolvable_states):
    for start in adj_list:
        if start in goal_states or not adj_list[start] or start in unsolvable_states:
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
        problem.linear_constraints.set_rhs(c_max_weight_name, 1 * max_weight)
        problem.linear_constraints.set_senses(c_max_weight_name, "L")
    problem.linear_constraints.set_coefficients(coefficients)

def populate_weight_selection_constraints(problem, num_features, max_weight):
    coefficients = []
    for f in range(0, num_features):
        w_name = get_weight_var(f)
        z_var = 'z_w_' + str(f)
        constraint_name = 'c_z_w_' + str(f)
        problem.indicator_constraints.add(
            indvar=z_var,
            complemented=0,
            rhs=0,
            sense="E",
            lin_expr=cplex.SparsePair(ind=[w_name], val=[1]),
            name=constraint_name)


def populate_absolute_value_weight_constraints(problem, num_features, max_weight):
    coefficients = []
    for f in range(0, num_features):
        w_name = get_weight_var(f)

        # x_+ constraint
        xplus_name = 'c_abs_w_plus_' + str(f)
        problem.linear_constraints.add(names=[xplus_name])
        coefficients.append((xplus_name, w_name, -1))
        coefficients.append((xplus_name, 'abs_w_' + str(f), 1))
        problem.linear_constraints.set_rhs(xplus_name, 0)
        problem.linear_constraints.set_senses(xplus_name, "G")

        # x_- constraint
        xminus_name = 'c_abs_w_minus_' + str(f)
        problem.linear_constraints.add(names=[xminus_name])
        coefficients.append((xminus_name, w_name, 1))
        coefficients.append((xminus_name, 'abs_w_' + str(f), 1))
        problem.linear_constraints.set_rhs(xminus_name, 0)
        problem.linear_constraints.set_senses(xminus_name, "G")

    problem.linear_constraints.set_coefficients(coefficients)

def extract_heuristic_parameters_from_cplex_solution(problem, nfeatures):
    """ Return a list of the tuples (i, w) that make up the potential heuristic, where i
    is the index of the feature and w is the learnt weight """
    # Cplex sometimes returns float values even if the variable is declared as
    # an integer, in particular when we use big-M instead of indicator
    # constraints.
    def make_weight_rounded(x):
        rounded_number = round(x,8)
        integer_number = int(round(x))
        if math.isclose(rounded_number, integer_number, abs_tol=0.00000009):
            return integer_number
        else:
            return rounded_number

    wvar_names = ((i, get_weight_var(i)) for i in range(0, nfeatures))
    feature_weights = ((i, wname, make_weight_rounded(problem.solution.get_values(wname))) for i, wname in wvar_names)
    nonzero_features = [(i, val) for i, var, val in feature_weights if val != 0]
    return nonzero_features


def create_toy_heuristic(features_per_state, parameters):
    """ Create a toy heuristic that works only with the feature values that we have already precomputed
    This will be useful e.g. for validation and debugging purposes.
    """

    def heuristic_function(s):
        state_values = features_per_state[s]
        h = 0
        for ind, fweight in parameters:
            h += fweight * state_values[ind]
        return h

    return heuristic_function


def report(parameters, heuristic, feature_names, feature_complexity, features_per_state, config):
    print("Concept-based potential heuristic found with a total of {} features:".format(len(parameters)))
    for i, val in parameters:
        print("\t{weight} · {feature} [k={k}, id={id}]".format(weight=val,
                                                               feature=config.feature_namer(feature_names[i]),
                                                               k=feature_complexity[i], id=i))

    sorted_state_ids = natural_sort(features_per_state.keys())
    with open(config.state_heuristic_filename, "w") as file:
        for s in sorted_state_ids:
            print("h({}) = {}".format(s, heuristic(s)), file=file)

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


def create_potential_heuristic_from_parameters(features, parameters):
    selected_features = [features[f_idx] for f_idx, _ in parameters]
    selected_weights = [f_weight for _, f_weight in parameters]
    return ConceptBasedPotentialHeuristic(list(zip(selected_features, selected_weights)))


class ConceptBasedPotentialHeuristic:
    def __init__(self, parameters):
        def transform(feature):
            if isinstance(feature, EmpiricalBinaryConcept):
                return ConceptCardinalityFeature(feature.c)
            return feature

        self.parameters = [(transform(feature), weight) for feature, weight in parameters]

    def value(self, model):
        h = 0
        for feature, weight in self.parameters:
            denotation = int(model.denotation(feature))
            delta = weight * denotation
            # logging.info("\t\tFeature \"{}\": {} = {} * {}".format(feature, delta, weight, denotation))
            h += delta

        # logging.info("\th(s)={} for state {}".format(h, state))
        return h

    def __str__(self):
        return " + ".join("{}·{}".format(weight, feature) for feature, weight in self.parameters)


def run(config, data, rng):
    max_weight = config.lp_max_weight

    transitions, adj_list = read_transition_file(config.transitions_filename)
    features_per_state, num_features = read_features_file(config.feature_matrix_filename)
    goal_states = read_state_set(config.goal_states_filename)
    unsolvable_states = read_state_set(config.unsolvable_states_filename)
    feature_complexity, feature_names = read_complexity_file(config.feature_info_filename)

    logging.info("Read {} transitions, {} features, {} goal states".
                 format(len(transitions), len(goal_states), num_features))

    logging.info("Populating model")
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)

    logging.info("Populating objective function")

    # See the code for options on obj function
    populate_obj_function_min_complexity(problem, num_features, max_weight, transitions, feature_complexity, goal_states)
    #populate_obj_function_max_nonselected_complexity(problem, num_features, max_weight, transitions, feature_complexity, goal_states)
    logging.info("Populating weight constraints")
    populate_weight_constraints(problem, transitions, features_per_state, goal_states, unsolvable_states, max_weight)
    logging.info("Populating dead-end constraints")
    populate_dead_end_constraints(problem, transitions, features_per_state, goal_states, unsolvable_states)
    logging.info("Populating y-constraints")
    populate_y_constraints(problem, adj_list, goal_states, unsolvable_states)
    logging.info("Populating M_w-constraints")

    # This function requires the min weighted complexity obj function. Otherwise, use the max weight constraints
    #populate_absolute_value_weight_constraints(problem, num_features, max_weight)
    #populate_weight_selection_constraints(problem, num_features, max_weight)
    populate_max_weight_constraints(problem, num_features, max_weight)

    logging.info("Writing file...")
    problem.write(config.lp_filename)
    logging.info("Total number of constraints in the MIP: %d", problem.linear_constraints.get_num() + problem.indicator_constraints.get_num())
    logging.info("Total number of variables in the MIP: %d", problem.variables.get_num())
    logging.info("Solving MIP...")

    # Below, some ways to limit resources. This allows suboptimal solutions (or no solution).
    # To set the timelimit:
    # problem.parameters.timelimit.set(100)
    # To set the limit of solutions found:
    # problem.parameters.mip.limits.solutions.set(10)
    # To set high numerical precision on, turn on the following:
    # problem.parameters.emphasis.numerical.set(1)
    problem.parameters.mip.strategy.probe.set(3)

    problem.solve()
    if problem.solution.is_primal_feasible() and problem.solution.is_dual_feasible():
        logging.info("Optimal solution found with value {}".format(problem.solution.get_objective_value()))

        parameters = extract_heuristic_parameters_from_cplex_solution(problem, len(feature_names))
        heuristic = create_toy_heuristic(features_per_state, parameters)

        report(parameters, heuristic, feature_names, feature_complexity, features_per_state, config)

        if config.validate_learnt_heuristic:
            # Run hill-climbing and make sure we find a goal. This won't work if we only have a sample of the
            # transition system, as in the incremental approach, since we might not have sampled a path to the goal.
            hill_climbing("s0", adj_list, heuristic, goal_states)

        # Return those values that we want to be persisted between different steps
        return ExitCode.Success, dict(learned_heuristic=create_potential_heuristic_from_parameters(
            data.features, parameters))
    elif problem.solution.is_primal_feasible():
        # TODO: if we limit the time or accept suboptimal solutions, we will
        # fall in this case. Is it right? How can we avoid it?
        logging.error("MIP is unbounded")
    elif problem.solution.is_dual_feasible():
        logging.error("MIP is unsolvable")
    else:
        logging.error("MIP was not solved. Unknown reason.")

    return ExitCode.MaxsatModelUnsat, dict()
    # raise CriticalPipelineError("MIP could not be solved")  # We'll add better error handling when necessary
