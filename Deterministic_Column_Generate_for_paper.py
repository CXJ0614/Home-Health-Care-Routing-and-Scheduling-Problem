# Home Health Care Routing & Scheduling - Single Service (SS)
# Deterministic column generation
#
# Data: A1 instance (hard-coded from the spreadsheet you provided)
# Model idea (paper's notation):
# - Each feasible day-route of one caregiver is a "column" r with cost:
#     travel(r) + beta * (overtime(r) + sum tardiness on route)
# - RMP (LP) chooses columns with:
#     * coverage: each patient i is covered exactly once
#     * headcount: each caregiver k selects exactly one route
# - Pricing (per caregiver): RCSP with time windows to find negative reduced-cost routes:
#     rc(r) = cost(r) - sum_{i in route} u_i - phi_k
# - After CG stops, we solve an Integer Master (IRMP) over the generated columns
#   to get a feasible 0/1 solution comparable with the paper.

import pulp
from collections import deque
import matplotlib.pyplot as plt
from itertools import permutations
# -----------------------------
# Global settings
# -----------------------------
beta = 100.0       # penalty weight for (tardiness + overtime) same as in the paper (P19)
max_stop = 60    # max patients per route in pricing

# -----------------------------
# 1) Sets & instance size
# -----------------------------
# patients 1..n ; node 0 = depot(start), node n+1 = sink(end)
n = 10
c = 3  # caregivers
q = 6  # service types (SS: each patient needs at most one)

# -----------------------------
# 2) Coordinates (A1)
# -----------------------------
XY = [
    [34, 53],  # 0 depot(start point)
    [42, 57],  # 1
    [5,  55],  # 2
    [20, 97],  # 3
    [94, 100], # 4
    [84, 41],  # 5
    [81, 99],  # 6
    [26, 17],  # 7
    [70, 1],   # 8
    [34, 69],  # 9
    [11, 11],  # 10
    [34, 53],  # 11 sink (end point same as depot)
]

# -----------------------------
# 3) Travel-time matrix Tij and cost cij (A1 "Travel time" table)
#    Size: (n+2) x (n+2), nodes: 0..n+1
# -----------------------------
T = [
#    0   1   2   3   4   5   6   7    8   9   10  11
    [ 0,  8, 29, 46, 76, 51, 65, 36,  63, 16, 47,  0],  # 0  depot -> j
    [ 8,  0, 37, 45, 67, 44, 57, 43,  62, 14, 55,  8],  # 1
    [29, 37,  0, 44, 99, 80, 87, 43,  84, 32, 44, 29],  # 2
    [46, 45, 44,  0, 74, 85, 61, 80, 108, 31, 86, 46],  # 3
    [76, 67, 99, 74,  0, 59, 13,107, 101, 67,121, 76],  # 4
    [51, 44, 80, 85, 59,  0, 58, 62,  42, 57, 78, 51],  # 5
    [65, 57, 87, 61, 13, 58,  0, 98,  98, 55,112, 65],  # 6
    [36, 43, 43, 80,107, 62, 98,  0,  46, 52, 16, 36],  # 7
    [63, 62, 84,108,101, 42, 98, 46,   0, 76, 59, 63],  # 8
    [16, 14, 32, 31, 67, 57, 55, 52,  76,  0, 62, 16],  # 9
    [47, 55, 44, 86,121, 78,112, 16,  59, 62,  0, 47],  # 10 (patient 10)
    [ 0,  8, 29, 46, 76, 51, 65, 36,  63, 16, 47,  0],  # 11 (sink)
]

# We also use T as travel cost c_ij in the objective:
travel_time = T
travel_cost  = T

# -----------------------------
# 4) Single-service needs delta_{is}  (A1 "Pat/Ser")
#    Each patient needs at most one service (SS).
# -----------------------------
req = [
    # S1 S2 S3 S4 S5 S6
    [0, 1, 0, 0, 0, 0],  # patient 1
    [0, 0, 1, 0, 0, 0],  # patient 2
    [1, 0, 0, 0, 0, 0],  # patient 3
    [0, 0, 0, 0, 0, 1],  # patient 4
    [0, 1, 0, 0, 0, 0],  # patient 5
    [0, 0, 0, 0, 1, 0],  # patient 6
    [0, 1, 0, 0, 0, 0],  # patient 7
    [0, 0, 1, 0, 0, 0],  # patient 8
    [0, 0, 1, 0, 0, 0],  # patient 9
    [1, 0, 0, 0, 0, 0],  # patient 10
]

# service_of[i] = the unique service s required by patient i (or -1 if none)
service_of = []
for i in range(n):
    s_idx = -1
    for s in range(q):
        if req[i][s] == 1:
            s_idx = s
            break
    service_of.append(s_idx)
print(service_of)

# -----------------------------
# 5) Service durations tilde t_{is} (A1 "P/SerTime")
# -----------------------------
dur = [
    [17,19,18,17,20,18],  # P1
    [18,16,18,19,17,20],  # P2
    [16,19,20,16,17,15],  # P3
    [16,20,19,18,18,16],  # P4
    [18,19,20,15,18,17],  # P5
    [15,17,19,17,17,20],  # P6
    [20,18,18,18,15,16],  # P7
    [16,15,15,17,18,18],  # P8
    [19,15,17,15,16,18],  # P9
    [18,18,16,20,19,18],  # P10
]

# -----------------------------
# 6) Caregiver skills Delta_{ks} (A1 "CgSkills")
# -----------------------------
skills = [
    [1,1,1,0,0,0],  # caregiver 0 : S1,S2,S3
    [0,0,0,0,1,1],  # caregiver 1 : S5,S6
    [0,1,1,0,0,0],  # caregiver 2 : S2,S3
]

# -----------------------------
# 7) Time windows (patient: [a_i,b_i], caregiver: [d_k,e_k])
# -----------------------------
cg_etw = [0, 0, 0]                 # d_k
cg_ltw = [600, 600, 600]           # e_k

etw = [207, 352,  78,  28, 117, 364, 457, 166, 329, 380]  # a_i
ltw = [327, 472, 198, 148, 237, 484, 577, 286, 449, 500]  # b_i

# -----------------------------
# 8) Some function
# -----------------------------
def can_serve(k, i):
    """Caregiver k can serve patient i if i needs some service and k is qualified for it."""
    s = service_of[i] # service required by patient i
    return (s != -1) and (skills[k][s] == 1)

def route_cost(travel, overtime, tard_sum):
    """Column cost = travel + beta*(overtime + tardiness_sum)."""
    return travel + beta*(overtime + tard_sum)

# -----------------------------
# 9) Initial columns: single-visit routes 0->i->end if feasible by time windows
# Column format:
# [k, stops[i...], starts[...], travel_cost, duration, overtime, tard_sum, alpha_tags[i...]]
# -----------------------------
def initial_columns():
    """
    Generate initial set of columns (routes) for the column generation process.
    Creates simple routes where each caregiver visits exactly one patient.
    Each route goes: depot -> patient -> sink (end point)
    """
    routes = []

    # For each caregiver
    for k in range(c):
        # Calculate caregiver's working duration
        Dk = cg_ltw[k] - cg_etw[k]

        # Try to create a route for each patient this caregiver can serve
        for i in range(n):
            # Skip if caregiver cannot serve this patient
            if not can_serve(k, i):
                continue

            # Calculate arrival time at patient location
            arr = cg_etw[k] + travel_time[0][i + 1]

            # Calculate actual start time (respecting earliest time window)
            start_i = max(arr, etw[i])

            # Calculate finish time after service
            finish_i = start_i + dur[i][service_of[i]]

            # Calculate total travel cost (depot->patient + patient->sink)
            tr_cost = travel_cost[0][i + 1] + travel_cost[i + 1][n + 1]

            # Calculate total route duration
            duration = (finish_i - cg_etw[k]) + travel_time[i + 1][n + 1]

            # Calculate overtime if any
            overtime = max(0.0, duration - Dk)

            # Calculate tardiness (how late we started serving the patient)
            tard_sum = max(0.0, start_i - ltw[i])

            # Create alpha tag (list of patients served in this route)
            alpha = [i]

            # Add route to initial set
            # Route format: [caregiver, patient_list, start_times, travel_cost,
            #               duration, overtime, tardiness_sum, alpha_tags]
            routes.append([k, [i], [start_i], tr_cost, duration, overtime, tard_sum, alpha])

    return routes


def _build_route_from_order(k, order):
    """
    Build a complete route column given a caregiver and visit order.

    Args:
        k: Caregiver index
        order: List of patient indices (0-based) in visiting sequence

    Returns:
        Route column in format [k, stops, starts, travel_cost, duration, overtime, tard_sum, alpha_tags]
        Returns None if route is infeasible (though current implementation doesn't check feasibility)
    """
    # Get caregiver shift parameters
    start_shift = cg_etw[k]
    shift_duration = cg_ltw[k] - cg_etw[k]

    # Initialize route tracking variables
    current_time = start_shift
    total_travel_cost = 0.0
    start_times = []
    last_patient = -1

    # Visit each patient in the specified order
    for patient_idx in order:
        # Calculate travel time from previous location
        if last_patient == -1:
            # First patient: travel from depot (node 0) to patient
            travel_time_cost = travel_time[0][patient_idx + 1]
        else:
            # Subsequent patients: travel from previous patient to current patient
            travel_time_cost = travel_time[last_patient + 1][patient_idx + 1]

        # Update total travel cost
        total_travel_cost += travel_time_cost

        # Calculate arrival time at patient location
        arrival_time = current_time + travel_time_cost

        # Determine actual service start time (respecting patient time window)
        service_start_time = max(arrival_time, etw[patient_idx])

        # Calculate service completion time
        service_end_time = service_start_time + dur[patient_idx][service_of[patient_idx]]

        # Record service start time and update current time
        start_times.append(service_start_time)
        current_time = service_end_time
        last_patient = patient_idx

    # Return to depot
    total_travel_cost += travel_cost[last_patient + 1][n + 1]

    # Calculate route metrics
    total_duration = (current_time - start_shift) + travel_time[last_patient + 1][n + 1]
    overtime_hours = max(0.0, total_duration - shift_duration)

    # Calculate total tardiness across all patients
    total_tardiness = sum(
        max(0.0, start_times[idx] - ltw[order[idx]])
        for idx in range(len(order))
    )

    # Return complete route information
    return [k, list(order), start_times, total_travel_cost, total_duration,
            overtime_hours, total_tardiness, list(order)]

###adding part####(for solve the problem of some patient not served)
def repair_columns_for_unique_patients(routes, max_perm=6, verbose=True):
    """
    Repair column pool before solving IRMP by creating complete daily routes for patients
    who can only be served by one specific caregiver.

    This ensures that patients with unique caregiver requirements are properly covered
    in the final integer solution.

    Args:
        routes: List of existing route columns to append to
        max_perm: Maximum number of patients in a group to enumerate all permutations
        verbose: Whether to print information about added routes

    Returns:
        Number of additional routes added to the pool
    """

    def caregivers_who_can_serve(patient_idx):
        """
        Get list of caregivers who can serve a specific patient based on skills.

        Args:
            patient_idx: Patient index

        Returns:
            List of caregiver indices who can serve this patient
        """
        service_type = service_of[patient_idx]
        qualified_caregivers = []

        # Check each caregiver's skills
        for caregiver_idx in range(c):
            if service_type != -1 and skills[caregiver_idx][service_type] == 1:
                qualified_caregivers.append(caregiver_idx)

        return qualified_caregivers

    # Group patients by their unique caregiver (patients who can only be served by one caregiver)
    unique_patients_by_caregiver = {caregiver_idx: [] for caregiver_idx in range(c)}

    # Identify patients with unique caregiver assignments
    for patient_idx in range(n):
        # Skip patients with no service requirement
        if service_of[patient_idx] == -1:
            continue

        # Find caregivers who can serve this patient
        qualified_caregivers = caregivers_who_can_serve(patient_idx)

        # If only one caregiver can serve, add to their unique patient list
        if len(qualified_caregivers) == 1:
            unique_caregiver = qualified_caregivers[0]
            unique_patients_by_caregiver[unique_caregiver].append(patient_idx)

    # Counter for added routes
    added_route_count = 0

    # Process each caregiver's unique patient list
    for caregiver_idx, patient_list in unique_patients_by_caregiver.items():
        # Skip if no unique patients or only one patient (already handled by initial columns)
        if len(patient_list) <= 1:
            continue

        best_route = None

        # For small groups, enumerate all possible visit orders
        if len(patient_list) <= max_perm:
            for order_tuple in permutations(patient_list):
                # Build route for this specific order
                route_candidate = _build_route_from_order(caregiver_idx, order_tuple)

                # Skip invalid routes
                if route_candidate is None:
                    continue

                # Keep track of the best (lowest cost) route
                if best_route is None or (
                        route_candidate[3] + beta * (route_candidate[5] + route_candidate[6]) <
                        best_route[3] + beta * (best_route[5] + best_route[6])
                ):
                    best_route = route_candidate
        else:
            # For large groups, use simple heuristic: sort by earliest time windows
            sorted_order = tuple(sorted(patient_list, key=lambda p: etw[p]))
            best_route = _build_route_from_order(caregiver_idx, sorted_order)

        # Add the best route to the column pool if found
        if best_route is not None:
            routes.append(best_route)
            added_route_count += 1

            # Print information if verbose mode is enabled
            if verbose:
                print(f"[REPAIR] add route for caregiver {caregiver_idx}: {best_route[1]}")

    return added_route_count
# -----------------------------
# 10) RMP: relax theta >= 0
#  Objective: sum_r cost(r) * theta_r
#  Constraints: coverage (each patient once), headcount (each caregiver one route)
#  Returns: LP obj, theta values, duals u_i (coverage), phi_k (headcount)
# -----------------------------
def solve_RMP(routes, verbose=False):
    # create problem
    prob = pulp.LpProblem("RMP_SS", pulp.LpMinimize)

    # create variable
    theta = [pulp.LpVariable(f"th_{r}", lowBound=0) for r in range(len(routes))]

    # objective(cost of column * column choose variable)
    prob += pulp.lpSum((rt[3] + beta * (rt[5] + rt[6])) * theta[r] for r, rt in
                       enumerate(routes))  # rt[3] = travel_cost; rt[5] = overtime; rt[6] = tard_sum

    # coverage constraints(each patient will be covered by one route)
    for i in range(n):
        if service_of[i] == -1:
            continue
        prob += pulp.lpSum(theta[r] for r, rt in enumerate(routes) if i in rt[7]) == 1, f"cover_{i}"

    # headcount (each caregiver exactly one route)
    for k in range(c):
        prob += pulp.lpSum(theta[r] for r, rt in enumerate(routes) if rt[0] == k) == 1, f"head_{k}"

    prob.solve(pulp.PULP_CBC_CMD(msg=verbose))  # solve model

    patient_dual = {}
    caregiver_dual = {}

    # Iterate through all constraints and extract dual values based on constraint names
    for constraint_name, constraint in prob.constraints.items():
        # Constraint name format: "cover_i" (coverage constraint for patient i) or "head_k" (route constraint for caregiver k)
        if constraint_name.startswith("cover_"):
            # Extract patient index i (e.g., "cover_3" -> 3)
            _, patient_id = constraint_name.split("_")  # Split string by "_"
            patient_id = int(patient_id)  # Convert to integer index
            # Store the dual value of this constraint (constraint.pi is the dual variable value)
            patient_dual[patient_id] = constraint.pi
        elif constraint_name.startswith("head_"):
            # Extract caregiver index k (e.g., "head_2" -> 2)
            _, caregiver_id = constraint_name.split("_")
            caregiver_id = int(caregiver_id)
            # Store the dual value of this constraint
            caregiver_dual[caregiver_id] = constraint.pi

    # obtain dual solution of RMP
    theta_val = [theta[r].value() for r in range(len(routes))]
    obj_val = pulp.value(prob.objective)

    return obj_val, theta_val, patient_dual, caregiver_dual

# -----------------------------
# 11) Pricing (per caregiver):
# Using label algorithm idea
# Reduced cost: rc(route r for caregiver k) =   [travel + beta*(overtime + tard_sum)] - sum_{i in route} u_i - phi_k
# -----------------------------
def pricing(patient_dual, caregiver_dual, max_cols_per_caregiver=5):
    """
       Pricing problem: find negative reduced-cost routes for each caregiver.

       This function implements a labeling algorithm to solve the pricing subproblem
       for each caregiver separately. It searches for feasible routes with negative
       reduced costs to add to the master problem.

       Args:
           patient_dual: Dictionary mapping patient index to its dual value (u_i)
           caregiver_dual: Dictionary mapping caregiver index to its dual value (phi_k)
           max_cols_per_caregiver: Maximum number of negative reduced cost columns
                                  to return per caregiver (default: 5)

       Returns:
           List of new route columns with negative reduced costs
       """
    new_routes = []

    # Pre-compute list of candidate patients each caregiver can serve
    candidate_patients = []
    for k in range(c):
        cand = []
        for i in range(n):
            if can_serve(k, i):
                cand.append(i)
        candidate_patients.append(cand)

    # Solve pricing problem for each caregiver
    for k in range(c):
        # Calculate shift duration for caregiver k
        Dk = cg_ltw[k] - cg_etw[k]
        start_shift = cg_etw[k]

        # Initialize queue for breadth-first search (labeling algorithm)
        # State format: (last_patient, current_time, visited_patients, start_times, travel_cost)
        Q = deque()
        Q.append((-1, start_shift, [], [], 0.0))  # -1 indicates starting from depot

        # Collect candidate columns with negative reduced costs
        cand_cols = []  # Store (reduced_cost, route) tuples

        # Breadth-first search to explore feasible routes
        while Q:
            # Pop state from queue
            last, tcur, stops, starts, tr_cost = Q.popleft()

            # Try to close the route (return to depot)
            if len(stops) > 0:
                # Get last visited patient
                last_i = stops[-1]
                # Calculate travel cost back to depot
                tr_back = travel_cost[last_i + 1][n + 1]
                # Calculate total route duration
                duration = (tcur - start_shift) + travel_time[last_i + 1][n + 1]
                # Calculate overtime (if any)
                overtime = max(0.0, duration - Dk)

                # Calculate total tardiness across all patients in route
                tard_sum = 0.0
                for idx in range(len(stops)):
                    i = stops[idx]
                    st_i = starts[idx]
                    tard_sum += max(0.0, st_i - ltw[i])

                # Calculate total travel cost
                total_travel = tr_cost + tr_back
                # Calculate route cost using objective function
                c_r = route_cost(total_travel, overtime, tard_sum)

                # Calculate reduced cost: rc(r) = cost(r) - sum_{i in route} u_i - phi_k
                u_sum = sum(patient_dual.get(i, 0.0) for i in stops)
                phi = caregiver_dual.get(k, 0.0)
                rc = c_r - u_sum - phi

                # If reduced cost is negative (within tolerance), add to candidate list
                if rc < -1e-6:
                    alpha = stops[:]
                    # Format route as column: [k, stops, starts, travel_cost, duration, overtime, tard_sum, alpha_tags]
                    r = [k, stops[:], starts[:], total_travel, duration, overtime, tard_sum, alpha]
                    cand_cols.append((rc, r))

            # Pruning: stop extending if maximum number of stops reached
            if len(stops) >= max_stop:
                continue

            # Extend current partial route by visiting one more patient
            for j in candidate_patients[k]:
                # Skip if patient already visited in current route
                if j in stops:
                    continue
                # Calculate travel time from last visited node to patient j
                travel = travel_time[0][j + 1] if last == -1 else travel_time[last + 1][j + 1]
                # Calculate arrival time at patient j
                arr = tcur + travel
                # Calculate actual start time (respecting patient's early time window)
                start_j = max(arr, etw[j])
                # Calculate finish time after serving patient j
                finish_j = start_j + dur[j][service_of[j]]
                # Add new state to queue for further exploration
                Q.append((j, finish_j, stops + [j], starts + [start_j], tr_cost + travel))

        # For current caregiver, add top negative columns to new_routes
        if cand_cols:
            # Sort by reduced cost (more negative is better)
            cand_cols.sort(key=lambda t: t[0])
            # Add up to max_cols_per_caregiver best columns
            for _, r in cand_cols[:max_cols_per_caregiver]:
                new_routes.append(r)

    return new_routes

# -----------------------------
# 12) Integer Master over generated columns (IRMP)
#     Gives a feasible 0/1 solution comparable to the paper's Z
# -----------------------------

def solve_integer_master(routes, verbose=False):
    """
    Solve Integer Master Problem (IRMP) to get a feasible 0/1 solution.

    This function solves the integer version of the master problem using the
    generated columns from column generation. It tries to use Gurobi first,
    and falls back to CBC if Gurobi is not available. The solution must be
    optimal, otherwise an error is raised.

    Args:
        routes: List of route columns generated during column generation
        verbose: Whether to print solver output

    Returns:
        Tuple of (objective_value, selected_route_indices)

    Raises:
        RuntimeError: If optimal solution is not found
    """
    # Create the integer programming problem
    prob = pulp.LpProblem("IRMP_SS", pulp.LpMinimize)

    # Create binary variables: th[r] = 1 if route r is selected, 0 otherwise
    th = [pulp.LpVariable(f"thb_{r}", lowBound=0, upBound=1, cat="Binary")
          for r in range(len(routes))]

    # Objective function: minimize total cost of selected routes
    # Cost includes travel cost and penalty for overtime and tardiness
    prob += pulp.lpSum((rt[3] + beta * (rt[5] + rt[6])) * th[r] for r, rt in enumerate(routes))

    # Coverage constraints: each patient must be covered exactly once
    for i in range(n):
        if service_of[i] == -1:  # Skip patients who don't need service
            continue
        prob += pulp.lpSum(th[r] for r, rt in enumerate(routes) if i in rt[7]) == 1

    # Headcount constraints: each caregiver must select exactly one route
    for k in range(c):
        prob += pulp.lpSum(th[r] for r, rt in enumerate(routes) if rt[0] == k) == 1

    # Try to solve with Gurobi first (more powerful solver)
    used = "GUROBI_CMD"
    try:
        prob.solve(pulp.GUROBI_CMD(msg=1 if verbose else 0,
                                   options=[("MIPGap", 0.0),  # No optimality gap
                                            ("IntFeasTol", 1e-9),  # Tight integer feasibility tolerance
                                            ("FeasibilityTol", 1e-9),  # Tight feasibility tolerance
                                            ("MIPFocus", 1)]))  # Focus on finding feasible solutions
    except Exception:
        # Fallback to CBC if Gurobi is not available
        used = "PULP_CBC_CMD"
        prob.solve(pulp.PULP_CBC_CMD(msg=1 if verbose else 0))

    # Get solution status
    status = pulp.LpStatus[prob.status]

    # If Gurobi failed to find optimal solution, try CBC
    if status != "Optimal" and used == "GUROBI_CMD":
        used = "PULP_CBC_CMD"
        prob.solve(pulp.PULP_CBC_CMD(msg=1 if verbose else 0))
        status = pulp.LpStatus[prob.status]

    # If still not optimal, collect diagnostic information and raise error
    if status != "Optimal":
        # Collect diagnostic information
        routes_by_caregiver = {k: 0 for k in range(c)}
        for route in routes:
            routes_by_caregiver[route[0]] += 1

        coverage_count = [0] * n
        for route in routes:
            for patient in route[7]:
                coverage_count[patient] += 1

        raise RuntimeError(f"[IRMP] status={status} (solver={used}), "
                           f"routes_by_k={routes_by_caregiver}, cover_cnt={coverage_count}")

    # Get objective value
    obj = pulp.value(prob.objective)

    # Collect indices of selected routes (where th[r] = 1)
    selected_routes = []
    for r, variable in enumerate(th):
        value = pulp.value(variable)
        if value is not None and value >= 0.999999:  # Using tolerance for floating point comparison
            selected_routes.append(r)

    return obj, selected_routes

# -----------------------------Column Generate Part Finished—---------------------------------------------------------




# -----------------------------
# Using Gurobi to get the optimal solution
# Direct MIP (PuLP + GUROBI_CMD) for Single-Service (SS)
# Returns: (obj_value, routes_by_k)
#   routes_by_k: list of length c; each element is a node sequence (e.g. [0, 3, 5, 11])
# -----------------------------
def solve_mip_with_pulp_gurobi(verbose=False):
    """
    Solve the problem directly as a MIP using Gurobi (or fallback to CBC)
    This creates a complete mathematical model with all constraints and solves it optimally
    """

    # Create optimization problem
    prob = pulp.LpProblem("HHCRSP_SS_MIP", pulp.LpMinimize)

    # Define index sets
    N0 = list(range(0, n + 1))  # Nodes 0 to n (excluding sink)
    N1 = list(range(1, n + 2))  # Nodes 1 to n+1 (excluding depot)
    K = list(range(c))  # Caregivers
    P = list(range(1, n + 1))  # Patients as nodes 1 to n

    # Patient service duration (use required service only; 0 if no service needed)
    dur_i = {}
    for i in P:
        if service_of[i - 1] == -1:
            dur_i[i] = 0
        else:
            dur_i[i] = dur[i - 1][service_of[i - 1]]

    # Large constant for big-M constraints
    BIGM_time = 10_000

    # Decision variables
    # x[i,j,k]: 1 if caregiver k travels from node i to node j, 0 otherwise
    x = {}
    for k in K:
        for i in N0:
            for j in N1:
                if j != i:  # Cannot travel from a node to itself
                    x[(i, j, k)] = pulp.LpVariable(f"x_{i}_{j}_{k}", lowBound=0, upBound=1, cat="Binary")

    # y[i,k]: 1 if patient i is served by caregiver k, 0 otherwise
    y = {}
    for i in P:
        for k in K:
            y[(i, k)] = pulp.LpVariable(f"y_{i}_{k}", lowBound=0, upBound=1, cat="Binary")

    # S[i,k]: Start time of service for patient i by caregiver k
    S = {}
    for i in P:
        for k in K:
            S[(i, k)] = pulp.LpVariable(f"S_{i}_{k}", lowBound=0)

    # E[k]: Completion time of caregiver k's route
    E = {}
    for k in K:
        E[k] = pulp.LpVariable(f"E_{k}", lowBound=0)

    # o[k]: Overtime of caregiver k
    o = {}
    for k in K:
        o[k] = pulp.LpVariable(f"o_{k}", lowBound=0)

    # v[i]: Tardiness of patient i (how late service starts)
    v = {}
    for i in P:
        v[i] = pulp.LpVariable(f"v_{i}", lowBound=0)

    # Objective function: minimize total travel cost + beta*(total tardiness + total overtime)
    travel_cost_term = pulp.lpSum(T[i][j] * x[(i, j, k)] for k in K for i in N0 for j in N1 if j != i)
    penalty_term = beta * (pulp.lpSum(v[i] for i in P) + pulp.lpSum(o[k] for k in K))
    prob += travel_cost_term + penalty_term

    # 1) Route start and end constraints
    # Each caregiver must start from depot (node 0) and end at sink (node n+1)
    for k in K:
        prob += pulp.lpSum(x[(0, j, k)] for j in N1) == 1, f"start_{k}"
        prob += pulp.lpSum(x[(i, n + 1, k)] for i in N0) == 1, f"end_{k}"

    # 2) Flow balance constraints at patient nodes
    # For each patient and caregiver: inbound flow equals outbound flow equals service indicator
    for k in K:
        for m in P:
            prob += pulp.lpSum(x[(i, m, k)] for i in N0 if i != m) == y[(m, k)], f"in_eq_{m}_{k}"
            prob += pulp.lpSum(x[(m, j, k)] for j in N1 if j != m) == y[(m, k)], f"out_eq_{m}_{k}"

    # 3) Coverage constraints
    # Each patient who needs service must be served exactly once; others must not be served
    for i in P:
        if service_of[i - 1] == -1:
            # Patient doesn't need service - fix all y[i,k] to 0
            for k in K:
                prob += y[(i, k)] == 0, f"noservice_{i}_{k}"
        else:
            # Patient needs service - must be served exactly once
            prob += pulp.lpSum(y[(i, k)] for k in K) == 1, f"cover_{i}"

    # 4) Skill constraints
    # A caregiver can only serve a patient if they have the required skill
    for i in P:
        s = service_of[i - 1]
        if s == -1:
            continue
        for k in K:
            prob += y[(i, k)] <= skills[k][s], f"skill_{i}_{k}"

    # 5) Time window and propagation constraints
    # a) Service start time must respect patient time windows
    for i in P:
        ai = etw[i - 1]  # Earliest start time
        bi = ltw[i - 1]  # Latest start time
        for k in K:
            # If patient is served, start time must be at least ai
            prob += S[(i, k)] >= ai * y[(i, k)], f"tw_low_{i}_{k}"
            # If patient is served, start time must be at most bi
            prob += S[(i, k)] <= (bi + BIGM_time * (1 - y[(i, k)])), f"tw_up_{i}_{k}"

    # b) Time propagation: i -> j on k
    # Ensure time consistency when traveling from one node to another
    for k in K:
        for i in range(0, n + 1):  # i=0 represents start of shift
            for j in P:
                if j == i:
                    continue
                # Calculate predecessor time: use d_k if i==0, otherwise S[i,k] + dur_i
                if i == 0:
                    from_time = cg_etw[k]
                else:
                    from_time = S[(i, k)] + dur_i[i]
                prob += S[(j, k)] >= from_time + T[i][j] - BIGM_time * (1 - x[(i, j, k)]), f"time_{i}_{j}_{k}"

    # c) Completion time and overtime constraints
    # E[k] must be at least the completion time of any patient served by caregiver k
    for k in K:
        for i in P:
            prob += E[k] >= S[(i, k)] + dur_i[i] + T[i][n + 1] - BIGM_time * (1 - y[(i, k)]), f"endlink_{i}_{k}"
        prob += E[k] >= cg_etw[k], f"end_lb_{k}"
        prob += o[k] >= E[k] - cg_ltw[k], f"ot_{k}"

    # d) Patient tardiness constraints
    # Tardiness is the amount by which service start time exceeds latest allowed time
    for i in P:
        bi = ltw[i - 1]
        for k in K:
            prob += v[i] >= S[(i, k)] - bi, f"tard_{i}_{k}"

    # Solve with selected solver
    try:
        gurobi_solver = pulp.GUROBI_CMD(msg=1 if verbose else 0)
        prob.solve(gurobi_solver)
    except Exception as e:
        print("[WARN] GUROBI_CMD not available, fallback to CBC:", e)
        prob.solve(pulp.PULP_CBC_CMD(msg=1 if verbose else 0))

    obj_val = pulp.value(prob.objective)

    # Extract routes: for each caregiver k, follow the x=1 arcs from 0 to n+1
    routes_by_k = []
    x_val = {}
    for key, v in x.items():
        if hasattr(v, 'varValue'):
            x_val[key] = v.varValue
        else:
            x_val[key] = v.value()

    for k in K:
        # Build adjacency list from solution
        next_of = {}
        for i in N0:
            for j in N1:
                if j == i:
                    continue
                if x_val.get((i, j, k), 0) > 0.5:
                    next_of[i] = j

        # Follow path from depot to sink
        route = [0]
        cur = 0
        visited_guard = set([0])
        while cur != n + 1 and cur in next_of:
            nxt = next_of[cur]
            route.append(nxt)
            if nxt in visited_guard:  # Safety check (should not happen)
                break
            visited_guard.add(nxt)
            cur = nxt
        routes_by_k.append(route)

    return obj_val, routes_by_k
# -----------------------------
# 14) Main loop
# -----------------------------

def solve():
    """
    Main solution loop implementing column generation algorithm
    """
    # Generate initial set of routes (one patient per caregiver)
    routes = initial_columns()
    print(f"[A1-SS] n={n}, q={q}, c={c}")
    print(f"initial columns: {len(routes)}")

    # Column generation loop
    for it in range(1, 51):
        # Solve restricted master problem (RMP) with current routes
        obj, theta_vals, cov_dual, head_dual = solve_RMP(routes, verbose=False)
        print(f"iter {it:02d} | RMP (LP) obj = {obj:.2f} | routes = {len(routes)}")

        # Solve pricing problem to find new routes with negative reduced cost
        new_routes = pricing(cov_dual, head_dual)

        # Stop if no improving columns found
        if not new_routes:
            print("  no improving columns -> stop")
            break

        # Add new routes to pool
        routes.extend(new_routes)

    # Calculate final LP lower bound after column generation
    obj_lp, _, _, _ = solve_RMP(routes, verbose=False)
    print(f"Final RMP (LP) lower bound = {obj_lp:.2f}")

    #
    repair_columns_for_unique_patients(routes, verbose=True)
    # Solve integer master problem to get feasible 0/1 solution
    obj_ip, chosen = solve_integer_master(routes, verbose=False)
    print(f"Integer master on generated columns = {obj_ip:.2f}")
    print(f"Chosen routes (indices): {chosen}")

    # Solve MIP directly using Gurobi (or CBC) for comparison
    mip_obj, mip_routes = solve_mip_with_pulp_gurobi(verbose=False)
    if mip_obj is not None:
        print(f"[MIP (GUROBI_CMD)] objective = {mip_obj:.2f}")
        print(f"[MIP] routes by caregiver: {mip_routes}")

    # Visualize results: Column Generation vs MIP
    try:
        # Convert selected CG routes to node sequences for plotting
        cg_routes_for_plot = routes_from_CG_selection(routes, chosen)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_routes(XY, cg_routes_for_plot, f"CG (IRMP) routes, obj={obj_ip:.2f}")
        plt.subplot(1, 2, 2)
        if mip_routes is not None:
            plot_routes(XY, list(enumerate(mip_routes)), f"MIP (Gurobi) routes, obj={mip_obj:.2f}")
        else:
            plt.axis('off')
            plt.text(0.5, 0.5, "MIP skipped", ha='center', va='center')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("[WARN] plotting skipped:", e)

##========================Plot====================
#plot function(plot function completely by gpt, no need to read)
def routes_from_CG_selection(routes, chosen_idx):
    pairs = []
    for r in chosen_idx:
        k, stops = routes[r][0], routes[r][1]
        seq = [0] + [i+1 for i in stops] + [n+1]
        pairs.append((k, seq))
    pairs.sort(key=lambda t: t[0])
    return pairs

def plot_routes(XY, routes_by_k, title):
    import matplotlib.lines as mlines
    plt.title(title)

    xs = [p[0] for p in XY]
    ys = [p[1] for p in XY]
    p_sc = plt.scatter(xs[1:n+1], ys[1:n+1], s=36, label="patients")
    d_sc = plt.scatter([xs[0]], [ys[0]], c='red',   s=64, label="depot")
    s_sc = plt.scatter([xs[n+1]], [ys[n+1]], c='green', s=64, label="sink")

    palette = ['tab:blue','tab:orange','tab:green','tab:red',
               'tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


    caregiver_handles = []
    for k, seq in routes_by_k:
        col = palette[k % len(palette)]

        for a, b in zip(seq[:-1], seq[1:]):
            xa, ya = XY[a]
            xb, yb = XY[b]

            plt.plot([xa, xb], [ya, yb], linewidth=2.2, color=col, alpha=0.95)

            plt.annotate('', xy=(xb, yb), xytext=(xa, ya),
                         arrowprops=dict(arrowstyle='->', color=col, lw=1.8,
                                         shrinkA=6, shrinkB=6, alpha=0.95))

        for order, node in enumerate(seq[1:-1], start=1):
            x, y = XY[node]
            plt.text(x+0.8, y+0.8, f"{node}#{order}", fontsize=9,
                     color=col, weight='bold')

        caregiver_handles.append(
            mlines.Line2D([], [], color=col, lw=3, label=f"caregiver {k}")
        )

    plt.legend(handles=caregiver_handles + [p_sc, d_sc, s_sc],
               loc="best", frameon=True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(alpha=0.25, linestyle='--', linewidth=0.6)

# Run
if __name__ == "__main__":
    solve()