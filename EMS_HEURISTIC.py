import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
import random
import math
from collections import deque
from EMS_BB import create_ems_model
 

# MYOPIC HEURISTIC AND VNS METAHEURISTIC 
#################################################################################################################################### 

# CHECK FEASIBILITY 
def check_feasibility(solution, model, binary_vars):

    original_bounds = [(v.LB, v.UB) for v in binary_vars]
    
    try:
        for i, value in enumerate(solution):
            binary_vars[i].LB = float(value)
            binary_vars[i].UB = float(value)
        
        model.update()
        model.optimize()
        
        return model.status == GRB.OPTIMAL
        
    finally:
        for i, (lb, ub) in enumerate(original_bounds):
            binary_vars[i].LB = lb
            binary_vars[i].UB = ub
        model.update()



# GET ASSIGNMENT COST
def getAssignmentCost(solution, model, binary_vars):

    original_bounds = [(v.LB, v.UB) for v in binary_vars]
    
    try:
        for i, value in enumerate(solution):
            binary_vars[i].LB = float(value)
            binary_vars[i].UB = float(value)
        
        model.update()
        model.optimize()
        
        return model.ObjVal if model.status == GRB.OPTIMAL else np.inf
        
    finally:
        for i, (lb, ub) in enumerate(original_bounds):
            binary_vars[i].LB = lb
            binary_vars[i].UB = ub
        model.update()

####################################################################################################################################

# MYOPIC HEURISTIC
def myopic_heuristic(P, model, binary_vars):
    print("************************    Running myopic heuristic    ************************\n\n")
    
    tempsol = np.zeros(P, dtype=int)
    best_var_assignment_cost = np.inf
    value_selected = 0
    
    # For each binary variable, try 0 and 1
    for i in range(P):
        best_var_assignment_cost = np.inf
        value_selected = 0
        for val in [0, 1]:   
            tempsol[i] = val
            if not check_feasibility(tempsol, model, binary_vars):
                continue
            cost = getAssignmentCost(tempsol, model, binary_vars)
            if best_var_assignment_cost > cost:  
                best_var_assignment_cost = cost
                value_selected = val
        
        tempsol[i] = value_selected
        solution = tempsol.copy()   
        print(f"{binary_vars[i].VarName} = {value_selected} (COST={best_var_assignment_cost:,.2f})")
    status = check_feasibility(solution, model, binary_vars)
    
    return status, solution, best_var_assignment_cost   



# VNS 
def VNS_algorithm(kmax, max_iterations, neighborhood_size, solution, solution_cost, P, model, binary_vars):
    print("************************    VNS METAHEURISTIC    ************************\n\n")
    k = 0 # index of neighborhood
    iterations = 0
    current_sol = np.array(solution, dtype=int)
    current_sol_cost = solution_cost
    # Create neighborhood
    neigborhoods = create_neighborhoods(kmax, P, neighborhood_size)
    while iterations < max_iterations:
        k = 0
        # Explore until k = kmax. k is the size of the neighborhood.
        while k < kmax:
            
            # Shaking current solution, to get a new point on search space
            new_sol, new_sol_cost = shaking(current_sol, neigborhoods[k], P, model, binary_vars)

            # Local search applied on the neighborhood
            new_sol, new_sol_cost = local_search_vns(neigborhoods[k], new_sol, P, model, binary_vars)
            
            # In case the new solution is better, keep it and set k=0
            if new_sol_cost < current_sol_cost:
                current_sol_cost = new_sol_cost
                current_sol = new_sol
                print(f"--IMPROVED BOUND: ITERATION {iterations}, k {k}, BEST EVALUATION {current_sol_cost:.5f}")
                k = 0
            # Otherwise, increase the size of the neighborhood
            else:
                k += 1
                
        # Optional: print progress
        if iterations % 10 == 0:
            print(f"ITERATION {iterations}, BEST EVALUATION {current_sol_cost:.5f}")
        iterations += 1
    print(f"LAST ITERATION {iterations-1}, BEST EVALUATION {current_sol_cost:.5f}")
    return current_sol, current_sol_cost



# Create k neighborhoods
def create_neighborhoods(k, P, neighborhood_size):
    vars = list(range(0, P))
    neighborhoods = []
    for i in range(k):
        
        random_sample = random.sample(vars, neighborhood_size)
        neighborhoods.append(random_sample)
    return neighborhoods



# Local search in VNS
def local_search_vns(neigborhood, solution, P, model, binary_vars):
    new_sol = np.copy(solution)
    for x in neigborhood:
        new_sol[x] = find_best_val(x, new_sol, P, model, binary_vars)

    sol_cost = getAssignmentCost(new_sol, model, binary_vars)
    return new_sol, sol_cost



# Shaking function in VNS
def shaking(solution, neighborhood, P, model, binary_vars):
    new_sol = np.copy(solution)
    i = 0
    while i < len(neighborhood):
        value_selected = random.randint(0, 1)
        new_sol[neighborhood[i]] = value_selected
        status = check_feasibility(new_sol, model, binary_vars)
        if status == True:   
            i += 1
    new_cost = getAssignmentCost(new_sol, model, binary_vars)
    return new_sol, new_cost



# find best value for variable x
def find_best_val(x, solution, P, model, binary_vars):
    test_sol = np.copy(solution)   
    heur_cost = np.inf
    value_selected = 0
    
    for val in [0, 1]:   
        test_sol[x] = val
        if not check_feasibility(test_sol, model, binary_vars):
            continue
        cost = getAssignmentCost(test_sol, model, binary_vars)
        if heur_cost > cost:
            heur_cost = cost
            value_selected = val
    return value_selected

####################################################################################################################################


# Set sense (min/max)
isMax = False

WARM_START = True
DEBUG_MODE = True

# Total number of nodes visited
nodes = 0

# Lower bound of the problem
lower_bound = -np.inf

# Upper bound of the problems 
upper_bound = np.inf

def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance


# A class 'Node' that holds information of a node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

# A simple function to print debugging info
def debug_print(node:Node = None, x_obj = None, sol_status = None):
        
        print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
        print(f"UB:{upper_bound}")
        print(f"LB:{lower_bound}")
        if node is not None:
            print(f"Brancing Var: {node.branching_var}")
        if node is not None:
            print(f"Child: {node.label}")
        if node is not None:
            print(f"Depth: {node.depth}")
        if x_obj is not None:
            print(f"Simplex Objective: {x_obj}")
        if sol_status is not None:
            print(f"Solution status: {sol_status}")

        print("\n\n--------------------------------------------------\n\n")


def check_depth_completion(depth, nodes_per_depth, best_bound_per_depth, lb, ub, isMax, DEBUG_MODE):
    stop = False
    
    # If all nodes in the current depth have been visited
    if nodes_per_depth[depth] == 0:
        if isMax:
            # Update Global Upper Bound
            ub = best_bound_per_depth[depth]
            if ub <= lb + 1e-6: # Check termination
                if DEBUG_MODE: print("Global UB hit LB (Level Completed). Stopping.")
                stop = True
        else:
            # Update Global Lower Bound
            lb = best_bound_per_depth[depth]
            if lb >= ub - 1e-6: # Check termination
                if DEBUG_MODE: print("Global LB hit UB (Level Completed). Stopping.")
                stop = True
                
    return stop, lb, ub

# Definition of the branch & bound algorithm.
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound

    # Create stack using deque() structure
    stack = deque()

    solutions = list()
    solutions_found = 0
    best_sol_idx = 0

    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    # Create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # ===============  Root node  ==========================

    if DEBUG_MODE:
        debug_print()
    
    # Solve relaxed problem
    model.optimize()

     # Check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth



    # Get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())
    
    # Get the objective value
    x_obj = model.ObjVal

    # Check if all variables have integer values (from the ones that are supposed to be integers)
    # If not, then select the first variable with a fractional value to be the one fixed
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            selected_var_idx = idx
            break



    # Found feasible solution.
    if vars_have_integer_vals:
        # If we have feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found
    
    # Otherwise update lower/upper bound for min/max respectively
    else:
        if isMax:
            upper_bound = x_obj    
        else:
            lower_bound = x_obj


    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    
    # Warm start simplex
    if WARM_START:
        # Retrieve vbasis and cbasis
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

    # Create lower bounds and upper bounds for the variables of the child nodes
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)
            


    # Create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # Create child nodes
    left_child = Node(left_ub, left_lb,root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # Add child nodes in stack
    stack.append(right_child)
    stack.append(left_child)

    # Solving sub problems
    # While the stack has nodes, continue solving
    while(len(stack) != 0):
        print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # Increment total nodes by 1
        nodes += 1

        # Get the child node on top of stack
        current_node = stack[-1]

        # Remove this node from stack
        stack.pop()

        # Increase the nodes visited for current depth
        nodes_per_depth[current_node.depth] -= 1


        # Warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        #print(f"LB: {current_node.lb}")
        #print(f"UB: {current_node.ub}")

        # Update the state of the model, passing the new lower bounds/upper bounds for the vars.
        # Basically, we only change the ub/lb for the branching variable. Another way is to introduce a new constraint (e.g. x_i <= ub).
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()    
        
        if DEBUG_MODE:
            debug_print()


        # Optimize the model
        model.optimize()

        # Check if the model was solved to optimality. If not then do not create child nodes.
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf
            
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 ** (i - current_node.depth)
            
            # if we reached the final node of a depth, then update the bounds
            stop, lower_bound, upper_bound = check_depth_completion(
                current_node.depth, nodes_per_depth, best_bound_per_depth, 
                lower_bound, upper_bound, isMax, DEBUG_MODE
            )
            if stop: 
                return solutions, best_sol_idx, solutions_found

        else:
            # Get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # Get the objective value
            x_obj = model.ObjVal

            # update best bound per depth if a better solution was found
            if isMax == True and x_obj > best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj
            elif isMax == False and x_obj < best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

        # If infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # Check if all variables have integer values (from the ones that are supposed to be integers)
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx
                break

        # Found feasible solution.
        # If integer solution found, then:
            # 1) - If solution improves incumbent, then store otherwise reject (optional)
            # If improves:
            # 2) - Update lb/ub for max/min respectively.
            # 3) - Check optimality condition lb=ub.
        if vars_have_integer_vals:
            if isMax:
                if lower_bound < x_obj:
                    lower_bound = x_obj
                    if abs(lower_bound - upper_bound) < 1e-6:  

                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1


                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found
                
                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1
                    
                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                    # if we reached the final node of a depth, then update the bounds
                    stop, lower_bound, upper_bound = check_depth_completion(
                        current_node.depth, nodes_per_depth, best_bound_per_depth, 
                        lower_bound, upper_bound, isMax, DEBUG_MODE
                    )
                    if stop: 
                        return solutions, best_sol_idx, solutions_found

                    
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue
               
            else:
                if upper_bound > x_obj:
                    upper_bound = x_obj
                    if abs(lower_bound - upper_bound) < 1e-6:  
                        
                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found
                
                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                    # if we reached the final node of a depth, then update the bounds
                    stop, lower_bound, upper_bound = check_depth_completion(
                        current_node.depth, nodes_per_depth, best_bound_per_depth, 
                        lower_bound, upper_bound, isMax, DEBUG_MODE
                    )
                    if stop: 
                        return solutions, best_sol_idx, solutions_found

                    
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue
            
            # do not branch further if is an equal solution
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 ** (i - current_node.depth)

            # if we reached the final node of a depth, then update the bounds
            stop, lower_bound, upper_bound = check_depth_completion(
                current_node.depth, nodes_per_depth, best_bound_per_depth, 
                lower_bound, upper_bound, isMax, DEBUG_MODE
            )
            if stop: 
                return solutions, best_sol_idx, solutions_found

            # Do not branch further if is an equal solution
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

        
        # If lb/ub for max/min respectively, is greater/less than x_obj then prune.
        # Here we accept x_obj = lb/ub (to potentially discover another solution with equal obj value) but this is optional. 
        # If we wanted to prune, the condition is: x_obj lower-equal (<=) to lower_bound    for a maximization problem.
        # For example:
        # if isMax:
        #   if (x_obj < lower_bound) or (abs(x_obj - lower_bound) < 1e-6):
        #       continue
        # else:
        #   if (x_obj > upper_bound) or (abs(x_obj - lower_bound) < 1e-6):
        #       continue

        
        if isMax:
  
            if x_obj < lower_bound:
                
                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                # if we reached the final node of a depth, then update the bounds
                stop, lower_bound, upper_bound = check_depth_completion(
                    current_node.depth, nodes_per_depth, best_bound_per_depth, 
                    lower_bound, upper_bound, isMax, DEBUG_MODE
                )
                if stop: 
                    return solutions, best_sol_idx, solutions_found

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue
        else:
            
            if x_obj > upper_bound:

                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 ** (i - current_node.depth)

                # if we reached the final node of a depth, then update the bounds
                stop, lower_bound, upper_bound = check_depth_completion(
                    current_node.depth, nodes_per_depth, best_bound_per_depth, 
                    lower_bound, upper_bound, isMax, DEBUG_MODE
                )
                if stop: 
                    return solutions, best_sol_idx, solutions_found

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        
        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")
        
        # Warm start simplex
        if WARM_START:
            # Retrieve vbasis and cbasis
            vbasis = model.getAttr("VBasis", model.getVars())
            cbasis = model.getAttr("CBasis", model.getConstrs())

        # Create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)


        # Create left and right branches  (e.g. set left: x = 0, right: x = 1 in a binary problem)
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # Create child nodes
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

        # Add child nodes in stack
        stack.append(right_child)
        stack.append(left_child)
    
    return solutions, best_sol_idx, solutions_found
      


# Main
if __name__ == "__main__":
    print("************************ MAIN ************************\n")
    
    model, ub, lb, integer_var, num_vars, vtypes, binary_vars = create_ems_model(8760)
    
    if model is None:
        print("MODEL NOT CREATED")
        exit(1)
    
    P = len(binary_vars)
    
    print(f"  BINARY: {sum(1 for vt in vtypes if vt == 'B')}")
    print(f"  INTEGER: {sum(1 for vt in vtypes if vt == 'I')}")
    print(f"  CONTINUOUS: {sum(1 for vt in vtypes if vt == 'C')}")
    print(f"  B&B USES: {np.sum(integer_var)}")
    
    
    # MYOPIC HEURISTIC
    print("\n************************ MYOPIC HEURISTIC ************************\n")
    
    myopic_status, myopic_solution, myopic_cost = myopic_heuristic(P, model, binary_vars)
    
    best_heuristic_cost = np.inf
    best_heuristic_solution = None
    best_heuristic_status = False
    
    if myopic_status and myopic_cost < best_heuristic_cost:
        best_heuristic_cost = myopic_cost
        best_heuristic_solution = myopic_solution.copy()
        best_heuristic_status = True
        print(f"\nMYOPIC HEURISTIC COST: {myopic_cost:,.2f})")
    
 
    # VNS METAHEURISTIC
    print("\n************************ VNS METAHEURISTIC ************************\n")
    if myopic_status:
        vns_sol, vns_cost = VNS_algorithm(kmax=2, max_iterations=10, neighborhood_size=2, solution=myopic_solution, solution_cost=myopic_cost, P=P, model=model, binary_vars=binary_vars)

        if vns_cost < best_heuristic_cost:
            best_heuristic_cost = vns_cost
            best_heuristic_solution = vns_sol.copy()
            best_heuristic_status = True

    
    # BRANCH & BOUND
    print("\n************************ EMS BRANCH & BOUND ************************\n")
    
    best_bound_per_depth = np.array([np.inf for _ in range(num_vars)])
    nodes_per_depth = [0 for i in range(num_vars)]
    
    if best_heuristic_status:
        upper_bound = best_heuristic_cost
        print(f">>> WARM START B&B WITH UB = {upper_bound:,.2f}\n")
    else:
        upper_bound = np.inf
        print(f">>> STARTING B&B FROM SCRATCH (UB = INF)\n")
    
    solutions, best_solution_index, solutions_found = branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth)
     

    # ΑΠΟΤΕΛΕΣΜΑΤΑ
    print(f"HEURISTIC FOUND: {best_heuristic_cost:,.2f} CHF/year")
    
    if solutions_found > 0:
        best_solution = solutions[np.argmin([s[1] for s in solutions])]
        solution_dict = {v.VarName: value for v, value in zip(model.getVars(), best_solution[0])}
        
        print("\n************************ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ ************************\n")
        
        try:
            print(f"Fuel Cell:     {solution_dict.get('x_gas_fc', 0)/1000:.2f} kW")
            print(f"PV:            {solution_dict.get('x_el_pv', 0)/1000:.2f} kWp")
            print(f"Solar Thermal: {solution_dict.get('x_th_st', 0)/650:.2f} m²")
            print(f"Heat Pump:     {solution_dict.get('x_el_hp', 0)/1000:.2f} kW")
            grid_kw = max(solution_dict.get(f'el_grid_in[{t}]', 0) for t in range(8760)) / 1000
            print(f"GRID:          {grid_kw:.2f} kW")
            print(f"Boiler:        {solution_dict.get('x_gas_boiler', 0)/1000:.2f} kW")
            print(f"Battery:       {solution_dict.get('y_el_battery', 0)/3.6e6:.2f} kWh")
            print(f"Tank height:   {solution_dict.get('y_h_tank', 0):.2f} m")
            #print(f"TREE DEPTH: {best_solution[2]}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        optimal_cost = best_solution[1]
        print(f"OPTIMAL COST: {optimal_cost:,.2f} CHF/year")
        
        if best_heuristic_status:
            gap = ((best_heuristic_cost - optimal_cost) / optimal_cost) * 100
            print(f"\nHEURISTIC:     {best_heuristic_cost:,.2f} CHF/year")
            print(f"B&B OPTIMAL:   {optimal_cost:,.2f} CHF/year")  
            print(f"GAP:           {gap:.4f}%")
            
    else:
        print("\nNO FEASIBLE SOLUTION FOUND IN B&B")