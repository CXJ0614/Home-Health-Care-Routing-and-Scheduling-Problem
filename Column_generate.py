import numpy as np
import pulp as pl

#choose solver
def get_solver(msg = False):
    try:
        return pl.GUROBI_CMD(msg = msg)
    except Exception:
        return pl.PULP_CBC_CMD(msg = msg)

# ======================= parameters =========================
# Exercise 2, p196
# first stage：min 7x1 + 11x2 + E(Q(x))
# second constraint：
#   y1 + 2y2 ≥ d1 - x1
#   y1 ≥ d2 - x2
#   y1, y2 ≥ 0；x1, x2 ∈ [0,10]
scenarios = [
    (0.5, 26.0, 16.0, 6.0, 12.0),   # scenarios1：(p, q1, q2, d1, d2)
    (0.5, 14.0, 24.0, 10.0, 4.0),  # scenarios2：(p, q1, q2, d1, d2)
]
C = np.array([7.0, 11.0])

# =============================== RMP ===============================
def solve_RMP(columns_scenario, solver):
    num_scenarios = len(scenarios) #calculate length of scenario

    #Create Problem
    prob = pl.LpProblem("RMP", pl.LpMaximize)

    #Create decision variable
    #define parameter lamda: lamda[scenario_index][col_index]
    #means the weight of scenario_index-th col_index-th
    lambda_list = []
    for scenarios_idx in range(num_scenarios):
        lambda_scenarios = [pl.LpVariable(f"lam_{scenarios_idx}_{col_idx}", lowBound= 0.0)
                            for col_idx in range(len(columns_scenario[scenarios_idx]))]
        lambda_list.append(lambda_scenarios)

    #create object function
    prob += pl.lpSum(
        lambda_list[scenarios_idx][col_idx] * columns_scenario[scenarios_idx][col_idx][1]
        for scenarios_idx in range(num_scenarios)
        for col_idx in range(len(columns_scenario[scenarios_idx]))
    )

    #add constraints
    #constraints1: sum(lambda * E[]) <= C
    for t in range(2):
        prob += pl.lpSum(
            lambda_list[scenarios_idx][col_idx] * columns_scenario[scenarios_idx][col_idx][0][t]
            for scenarios_idx in  range(num_scenarios)
            for col_idx in range(len(columns_scenario[scenarios_idx]))
        ) <= C[t]

    #constraints2: sum(lambda) = 1
    prob += pl.lpSum(
        lambda_list[scenarios_idx][col_idx]
        for scenarios_idx in  range(num_scenarios)
        for col_idx in range(len(columns_scenario[scenarios_idx]))
    ) == 1.0

    #solve the model
    prob.solve(solver)
    #check status
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError("RMP not optimal:" + pl.LpStatus)

    #obtain result
    z = float(pl.value(prob.objective)) # z = RMP obkject
    lambda_value = [
        [float(pl.value(lambda_list[scenarios_idx][col_idx])) for col_idx in range(len(lambda_list[scenarios_idx]))]
        for scenarios_idx in range(num_scenarios)
    ]

    #obtain the dual solution of RMP (for Pricing Problem)
    u, v = solve_RMP_dual(columns_scenario, solver)

    return z, lambda_value, u, v

#solve RMP dual problem
def solve_RMP_dual(columns_scenario, solver):
    #creat problem
    prob = pl.LpProblem("RMP_dual", pl.LpMinimize)

    #create dual variable
    u1 = pl.LpVariable("u1", lowBound= 0.0, upBound=10)
    u2 = pl.LpVariable("u2", lowBound= 0.0, upBound=10)
    v = pl.LpVariable("v", lowBound= None)

    #create objective function
    prob += C[0] * u1 + C[1] *u2 + v

    #create constraints
    #For each column (E, e) s.t. u1 * E[0] + u2 * E[1] + v ≥ e
    for scenarios_idx in range(len(columns_scenario)):
        for (E,e) in columns_scenario[scenarios_idx]:
            prob += u1 * E[0] + u2 * E[1] + v >= e

    #solve the dual problem
    prob.solve(solver)
    # check status
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"RMP_dual not optimal: {pl.LpStatus[prob.status]}")

    #obtain values
    u = np.array([float(pl.value(u1)),float(pl.value(u2))])
    v_val = float(pl.value(v))

    return u, v_val

#=====================Subproblem (Pricing Problem)============================
def pricing_problem(q1, q2, d1, d2, u, v, p, solver):
    #create pricing problem
    prob = pl.LpProblem("pricing problem", pl.LpMaximize)

    #create variable
    pi1 = pl.LpVariable("pi1", lowBound= 0.0)
    pi2 = pl.LpVariable("pi2", lowBound= 0.0)

    #create objective function
    prob += p * ((d1- u[0]) * pi1 + (d2 - u[1]) * pi2) - v

    #add constraints
    prob += pi1 + pi2 <= q1
    prob += 2 * pi1 <= q2

    # solve problem
    prob.solve(solver)
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Pricing_Problem not optimal: {pl.LpStatus[prob.status]}")

    # obtain result
    reduce_cost = float(pl.value(prob.objective))
    pi = np.array([float(pl.value(pi1)), float(pl.value(pi2))])
    E = p * pi.copy()  #New column parameter
    e = p * (d1 * pi[0] + d2 * pi[1])  #constant of new column

    return reduce_cost, E, e, pi

#===========================main loop===========================================
def column_generation(max_iter = 100, tol = 1e-9, msg = False):
    solver = get_solver(msg = msg)
    num_scenarios = len(scenarios)

    #step 0: initialization
    column_scenarios = [[] for _ in range(num_scenarios)]
    #add a column(avoid no feasible initial solution for RMP)
    column_scenarios[0].append((np.array([0.0, 0.0]), 0.0))

    #step 1: for each scenario generate initial column
    for scenarios_idx in range(num_scenarios):
        p, q1, q2, d1, d2 = scenarios[scenarios_idx]
        #use the initial dual variable u=0, v=0, generate the first column
        reduce_cost, E, e, pi = pricing_problem(q1, q2, d1, d2, u=np.zeros(2), v=0.0, p=p, solver=solver)
        column_scenarios[scenarios_idx].append((E,e))
        print(f"Column of Initial scenarios{scenarios_idx+1}: E={E}, e={e:.4f}")

    #step 2: generate new column
    for iter in range(max_iter):
        print(f"\n===== iteration {iter} =====")

        #2.1 solve the current RMP, obtain obj and u,v
        z, lam_val, u, v = solve_RMP(column_scenarios, solver)
        print(f"RMP object = {z:.6f}，dual variable u = {u}，v = {v:.6f}")
        #print each weight
        for scenarios_idx in range(num_scenarios):
            print(f"Scenarios{scenarios_idx+1} weight of column{[f'{val:.4f}' for val in lam_val[scenarios_idx]]}(column num:{len(column_scenarios[scenarios_idx])})")

        #2.2 solve the pricing problem for each scenario
        any_new_column = False

        for scenarios_idx in range(num_scenarios):
            #get parameters
            p, q1, q2, d1, d2 = scenarios[scenarios_idx]
            #solve the problem
            reduce_cost, E, e, pi = pricing_problem(q1, q2, d1, d2, u, v, p, solver)
            print(f"Scenarios{scenarios_idx+1}: Reduce Cost{reduce_cost}, Dual Variable pi = {pi}")

            #if rc > tol, add new column
            if reduce_cost > tol:
                column_scenarios[scenarios_idx].append((E,e))
                any_new_column = True
                print(f"-> add new column: E={E}, e={e}")

        #2.3 judge convenge
        if not any_new_column:
            print("No add new column")
            return z,u,v,column_scenarios
    raise RuntimeError("Not convergence")

#==============main=================================================
if __name__ == "__main__":
    z_star, u_star, v_star, all_columns = column_generation(msg=False)

    # print final result
    print("\n=== final result ===")
    print(f"Objective = {z_star:.6f}")
    print(f"Dual variable u = {u_star}，v = {v_star:.6f}")
    #print all column
    for scenario_idx, columns in enumerate(all_columns, 1):
        print(f"\nScenario{scenario_idx} generate column：")
        for col_idx, (E, e) in enumerate(columns, 1):
            print(f"  column{col_idx}：E={E}，e={e:.6f}")