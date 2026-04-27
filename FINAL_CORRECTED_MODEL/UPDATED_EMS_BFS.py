# EMS BRANCH & BOUND WITH BEST-FIRST SEARCH
from gurobipy import GRB
import numpy as np
import time
import random
import heapq
import itertools
from UPDATED_EMS_BB import create_ems_model, load_data


# MYOPIC HEURISTIC AND VNS METAHEURISTIC 
#################################################################################################################################### 

# CHECK FEASIBILITY 
def check_feasibility(solution, model, binary_vars):

    # Continuous (0, 1), (0, 1)... for all binary vars
    bounds = [(i.LB, i.UB) for i in binary_vars] 
    
    # 0 or 1 according to tempsol,new_sol,test_sol
    try:
        for i, value in enumerate(solution): 
            binary_vars[i].LB = float(value)
            binary_vars[i].UB = float(value)
        
        model.update() # Update the model with the new bounds
        model.optimize() # Solve the model with the fixed binary variables  
        
        # Return True if found feasible solution
        return model.status == GRB.OPTIMAL 
    
    # Restore the bounds for the next calls         
    finally:
        for i in range(len(binary_vars)):  
            lb = bounds[i][0]
            ub = bounds[i][1]
            binary_vars[i].LB = lb
            binary_vars[i].UB = ub
        model.update()  



# GET ASSIGNMENT COST
def getAssignmentCost(solution, model, binary_vars):

    bounds = [(i.LB, i.UB) for i in binary_vars] 
    
    try:
        for i, value in enumerate(solution): 
            binary_vars[i].LB = float(value)
            binary_vars[i].UB = float(value)
        
        model.update() 
        model.optimize() 
        
        return model.ObjVal if model.status == GRB.OPTIMAL else np.inf  
        
         
    finally:
        for i in range(len(binary_vars)): 
            lb = bounds[i][0]
            ub = bounds[i][1]
            binary_vars[i].LB = lb
            binary_vars[i].UB = ub
        model.update() 

####################################################################################################################################

# MYOPIC HEURISTIC
# For each binary variable, try 0 and 1
def myopic_heuristic(P, model, binary_vars):
    print("************************    Running myopic heuristic    ************************\n\n")

    tempsol = np.zeros(P, dtype=int)
    best_var_assignment_cost = np.inf
    value_selected = 0
    
    for i in range(P):
        best_var_assignment_cost = np.inf
        value_selected = 0
        for val in [0, 1]:   
            tempsol[i] = val
            cost = getAssignmentCost(tempsol, model, binary_vars)
            if cost == np.inf:
                continue
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
        # Explore until k = kmax. kmax is the number of the neighborhoods.
        while k < kmax:
            
            # Shaking current solution, to get a new point on search space
            new_sol, new_sol_cost = shaking(current_sol, neigborhoods[k], P, model, binary_vars)

            # Local search applied on the neighborhood
            new_sol, new_sol_cost = local_search_vns(neigborhoods[k], new_sol, model, binary_vars)
            
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
# Each neighborhood contains a random sample of variables of size neighborhood_size
def create_neighborhoods(k, P, neighborhood_size):
    vars = list(range(0, P))
    neighborhoods = []
    for i in range(k):
        
        random_sample = random.sample(vars, neighborhood_size)
        neighborhoods.append(random_sample)
    return neighborhoods



# Local search in VNS
# for each variable in the neighborhood, find the best value (0 or 1)
def local_search_vns(neigborhood, solution, model, binary_vars):
    new_sol = np.copy(solution)
    for x in neigborhood:
        new_sol[x] = find_best_val(x, new_sol, model, binary_vars)

    sol_cost = getAssignmentCost(new_sol, model, binary_vars)
    return new_sol, sol_cost



# Shaking function in VNS
# randomly change the values of the variables in the neighborhood
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
# try 0 and 1 and select the one with the best cost
def find_best_val(x, solution, model, binary_vars):
    test_sol = np.copy(solution)   
    heur_cost = np.inf
    value_selected = 0
    
    for val in [0, 1]:   
        test_sol[x] = val
        cost = getAssignmentCost(test_sol, model, binary_vars)
        if cost == np.inf:
            continue
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


# Function to find the most fractional variable
def most_fractional_variable(x_candidate, integer_var):
    most_fractional = 0.0
    selected_var_idx = -1
    all_integer = True
    
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            all_integer = False
            frac = abs(x_candidate[idx] - round(x_candidate[idx]))
            if frac > most_fractional:
                most_fractional = frac
                selected_var_idx = idx
    
    return all_integer, selected_var_idx


# Definition of the branch & bound algorithm (Best-First Search).
def branch_and_bound(model, ub, lb, integer_var, vbasis=[], cbasis=[], depth=0, tol=1e-6, incumbent_obj=np.inf):
    global nodes, lower_bound, upper_bound
    
    upper_bound = incumbent_obj

    # Priority queue
    pq = []
    counter = itertools.count()  # To break the tie for nodes with the same priority

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
    # Count root as a node.
    nodes += 1
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
            return [], -np.inf, depth



    # Get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())
    
    # Get the objective value
    x_obj = model.ObjVal

    # Check if all variables have integer values and find the most fractional variable
    vars_have_integer_vals, selected_var_idx = most_fractional_variable(x_candidate, integer_var)


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

    # Add child nodes to priority queue with parent's relaxation objective as priority
    if isMax:
        priority = -x_obj
    else:
        priority = x_obj    
    heapq.heappush(pq, (priority, next(counter), left_child))
    heapq.heappush(pq, (priority, next(counter), right_child))

    # Solving sub problems
    # While the priority queue has nodes, continue solving
    while len(pq) != 0:
        print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # Increment total nodes by 1
        nodes += 1

        # Get the node with best (lowest - minimization) bound from priority queue
        _, _, current_node = heapq.heappop(pq)


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

        else:
            # Get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # Get the objective value
            x_obj = model.ObjVal

        # If infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # Check if all variables have integer values and find the most fractional variable
        vars_have_integer_vals, selected_var_idx = most_fractional_variable(x_candidate, integer_var)

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
                    if abs(lower_bound - upper_bound) < tol:  

                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < tol) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1


                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found
                
                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= tol) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue
               
            else:
                if upper_bound > x_obj:
                    upper_bound = x_obj
                    if abs(lower_bound - upper_bound) < tol:  
                        
                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) <= tol) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found
                
                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= tol) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue
            
            # Do not branch further if is an equal solution
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

        
        # If lb/ub for max/min respectively, is greater/less than x_obj then prune.
        # Here we accept x_obj = lb/ub (to potentially discover another solution with equal obj value) but this is optional. 
        # If we wanted to prune, the condition is: x_obj lower-equal (<=) to lower_bound    for a maximization problem.
        # For example:
        # if isMax:
        #   if (x_obj < lower_bound) or (abs(x_obj - lower_bound) < tol):
        #       continue
        # else:
        #   if (x_obj > upper_bound) or (abs(x_obj - lower_bound) < tol):
        #       continue

        
        if isMax:
  
            if x_obj < lower_bound:

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue
        else:
            
            if x_obj > upper_bound:

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

        # Add child nodes to priority queue with current relaxation objective as priority
        priority = x_obj if not isMax else -x_obj
        heapq.heappush(pq, (priority, next(counter), left_child))
        heapq.heappush(pq, (priority, next(counter), right_child))
    
    return solutions, best_sol_idx, solutions_found



# SOLVER
def solve_instance(inputs: dict) -> dict:

    global nodes, lower_bound, upper_bound
    nodes = 0
    lower_bound = -np.inf
    upper_bound = np.inf

    T = inputs["T"]

    model, ub, lb, integer_var, num_vars, vtypes, binary_vars = create_ems_model(T=T, I_t=inputs["I_t"], Tamb=inputs["Tamb"], 
            Tcoll=inputs["Tcoll"], L_electricity=inputs["L_electricity"], L_dhw=inputs["L_dhw"], L_sph=inputs["L_sph"],)

    P = len(binary_vars)
    
    print(f"  BINARY: {sum(1 for vt in vtypes if vt == 'B')}")
    print(f"  INTEGER: {sum(1 for vt in vtypes if vt == 'I')}")
    print(f"  CONTINUOUS: {sum(1 for vt in vtypes if vt == 'C')}")
    print(f"  B&B USES: {np.sum(integer_var)}")


    # MYOPIC HEURISTIC
    print("\n************************ MYOPIC HEURISTIC ************************\n")
    
    start_time_myopic = time.time()
    myopic_status, myopic_solution, myopic_cost = myopic_heuristic(P, model, binary_vars)
    end_time_myopic = time.time()
    myopic_time = end_time_myopic - start_time_myopic
    print(f"\n--- MYOPIC TIME: {myopic_time:.2f} seconds ---\n")
    
    heuristic_cost = np.inf
    heuristic_solution = None
    heuristic_status = False
    
    if myopic_status and myopic_cost < heuristic_cost:
        heuristic_cost = myopic_cost
        heuristic_solution = myopic_solution.copy()
        heuristic_status = True
        print(f"\nMYOPIC HEURISTIC COST: {myopic_cost:,.2f}")
    
 
    # VNS METAHEURISTIC
    print("\n************************ VNS METAHEURISTIC ************************\n") 
    vns_time = 0.0
    if myopic_status:
        start_time_vns = time.time()
        vns_sol, vns_cost = VNS_algorithm(kmax=3, max_iterations=2, neighborhood_size=2, solution=myopic_solution, solution_cost=myopic_cost, P=P, model=model, binary_vars=binary_vars)
        end_time_vns = time.time()
        vns_time = (end_time_vns - start_time_vns) + myopic_time
        print(f"\n--- VNS TIME: {vns_time:.2f} seconds ---\n")

        if vns_cost < heuristic_cost:
            heuristic_cost     = vns_cost
            heuristic_solution = vns_sol.copy()
            heuristic_status   = True

    # BRANCH & BOUND
    print("\n************************ EMS BRANCH & BOUND ************************\n")

    if heuristic_status:
        upper_bound = heuristic_cost
        print(f">>> WARM START B&B WITH UB = {upper_bound:,.2f}\n")
    else:
        upper_bound = np.inf
        print(f">>> STARTING B&B FROM SCRATCH (UB = INF)\n")

    start_time_bb = time.time()
    solutions, best_solution_index, solutions_found = branch_and_bound(model, ub, lb, integer_var, incumbent_obj=upper_bound)
    end_time_bb = time.time()
    bb_time = (end_time_bb - start_time_bb) + vns_time
    print(f"\n--- B&B TIME: {bb_time:.2f} seconds ---\n")


    # ΑΠΟΤΕΛΕΣΜΑΤΑ
    # myopic cost
    print(f"\nMYOPIC HEURISTIC FOUND: {myopic_cost:,.2f} CHF/year")
    print(f"META-HEURISTIC FOUND: {heuristic_cost:,.2f} CHF/year")

    stats = {"nodes_visited": nodes, "solutions_found": solutions_found, "myopic_time_s": myopic_time,
        "vns_time_s": vns_time, "bb_time_s": bb_time, "heuristic_cost": heuristic_cost, "heuristic_status": heuristic_status}

    if solutions_found > 0:
        best_solution = solutions[np.argmin([s[1] for s in solutions])]
        solution_dict = {v.VarName: value for v, value in zip(model.getVars(), best_solution[0])}

        print("\n************************ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ ************************\n")
        try:
            print(f"Fuel Cell Capacity:         {solution_dict.get('x_gas_fc',     0)/1000:.2f} kW")
            print(f"PV Capacity:                {solution_dict.get('x_el_pv',      0)/1000:.2f} kWp")
            print(f"Solar Thermal Area:         {solution_dict.get('x_th_st',      0)/650:.2f} m²")
            print(f"Heat Pump Capacity:         {solution_dict.get('x_el_hp',      0)/1000:.2f} kW")
            grid_kw = max(solution_dict.get(f'el_grid_in[{t}]', 0) for t in range(T)) / 1000
            print(f"Grid Connection Size:       {grid_kw:.2f} kW")
            print(f"Gas Boiler Capacity:        {solution_dict.get('x_gas_boiler', 0)/1000:.2f} kW")
            print(f"Battery Capacity:           {solution_dict.get('y_el_battery', 0)/3.6e6:.2f} kWh")
            print(f"Thermal Tank height:        {solution_dict.get('y_h_tank',     0):.2f} m")
            print(f"Thermal Tank Volume:        {solution_dict.get('y_h_tank', 0) * np.pi * (0.80/2.0)**2 * 1000:.0f} Liters")

            best_bb_cost = best_solution[1]
            
            dT_vals = [solution_dict.get(f'dT[{t}]', 0) for t in range(T)]
            penalty_hours = sum(1 for v in dT_vals if v > 1e-6)
            total_penalty_cost = 0.1 * sum(dT_vals)  # c_T * sum(dT)
            print(f"\n--- COMFORT ---")
            print(f"Hours with penalty (dT>0):  {penalty_hours} / {T}")
            print(f"Total penalty cost:         {total_penalty_cost:,.2f} CHF/year")
            print(f"Clean energy cost:          {best_bb_cost - total_penalty_cost:,.2f} CHF/year")

            print(f"\n--- TIMES ---")
            print(f"Myopic Heuristic Time:      {myopic_time:.2f} seconds")
            print(f"VNS Metaheuristic Time:     {vns_time:.2f} seconds")
            print(f"Branch & Bound Time:       {bb_time:.2f} seconds")
            
            #print(f"TREE DEPTH: {best_solution[2]}")
        except Exception as e:
            print(f"ERROR: {e}")

     
        if heuristic_status:
            gap = ((heuristic_cost - best_bb_cost) / best_bb_cost) * 100
            print(f"B&B OPTIMAL COST:   {best_bb_cost:,.2f} CHF/year")
            print(f"GAP BETWEEN HEURISTIC AND BB:  {gap:.4f}%")
            print(f"NODES VISITED: {nodes}")

        result = {"bb_cost": best_bb_cost, "stats": stats}
    else:
        if heuristic_status:
            print("\nNO IMPROVING INTEGER SOLUTION FOUND IN B&B")
            print("HEURISTIC RESULT")

            bounds = [(v.LB, v.UB) for v in binary_vars]
            try:
                for i, val in enumerate(heuristic_solution):
                    binary_vars[i].LB = float(val)
                    binary_vars[i].UB = float(val)
                model.update()
                model.optimize()
                heuristic_solution_dict = {v.VarName: v.X for v in model.getVars()}
            finally:
                for i, v in enumerate(binary_vars):
                    v.LB = bounds[i][0]
                    v.UB = bounds[i][1]
                model.update()

            print("\n************************ ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ ************************\n")
            try:
                print(f"Fuel Cell Capacity:         {heuristic_solution_dict.get('x_gas_fc',     0)/1000:.2f} kW")
                print(f"PV Capacity:                {heuristic_solution_dict.get('x_el_pv',      0)/1000:.2f} kWp")
                print(f"Solar Thermal Area:         {heuristic_solution_dict.get('x_th_st',      0)/650:.2f} m²")
                print(f"Heat Pump Capacity:         {heuristic_solution_dict.get('x_el_hp',      0)/1000:.2f} kW")
                grid_kw = max(heuristic_solution_dict.get(f'el_grid_in[{t}]', 0) for t in range(T)) / 1000
                print(f"Grid Connection Size:       {grid_kw:.2f} kW")
                print(f"Gas Boiler Capacity:        {heuristic_solution_dict.get('x_gas_boiler', 0)/1000:.2f} kW")
                print(f"Battery Capacity:           {heuristic_solution_dict.get('y_el_battery', 0)/3.6e6:.2f} kWh")
                print(f"Thermal Tank Height:        {heuristic_solution_dict.get('y_h_tank',     0):.2f} m")
                print(f"Thermal Tank Volume:        {heuristic_solution_dict.get('y_h_tank', 0) * np.pi * (0.80/2.0)**2 * 1000:.0f} Liters")

                dT_vals = [heuristic_solution_dict.get(f'dT[{t}]', 0) for t in range(T)]
                penalty_hours = sum(1 for v in dT_vals if v > 1e-6)
                total_penalty_cost = 0.1 * sum(dT_vals)
                print(f"\n--- COMFORT ---")
                print(f"Hours with penalty (dT>0):  {penalty_hours} / {T}")
                print(f"Total penalty cost:         {total_penalty_cost:,.2f} CHF/year")
                print(f"Clean energy cost:          {heuristic_cost - total_penalty_cost:,.2f} CHF/year")
            except Exception as e:
                print(f"ERROR printing solution: {e}")

            print(f"\n--- TIMES ---")
            print(f"Myopic Heuristic Time:      {myopic_time:.2f} seconds")
            print(f"VNS Metaheuristic Time:     {vns_time:.2f} seconds")
            print(f"Branch & Bound Time:        {bb_time:.2f} seconds")
            print(f"NODES VISITED: {nodes}")
            print(f"\nHEURISTIC INCUMBENT COST: {heuristic_cost:,.2f} CHF/year")
            result = {"bb_cost": heuristic_cost, "stats": stats}
        else:
            print("\nNO FEASIBLE SOLUTION FOUND IN B&B")
            result = {"bb_cost": np.inf, "stats": stats}

    return result       

 
# MAIN
if __name__ == "__main__":
    import os

    data = load_data(location="THESS", T=8760, data_directory = os.path.expanduser("~/DB_WORKSPACE/HOUSE1"))

    result = solve_instance(inputs=data)
