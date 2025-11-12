import numpy as np
import pulp as pl

#choose solver
def get_solver(msg = False):
    try:
        return pl.GUROBI_CMD(msg = msg)
    except Exception:
        return pl.PULP_CBC_CMD(msg = msg)

#solve the linear master problem
def master_problem_multi(cuts_k, p_list, solver):
    #calculate the length K
    K = len(p_list)
    #create problem
    prob = pl.LpProblem("master",pl.LpMinimize)

    #create decision variable
    x1 = pl.LpVariable("x1",lowBound=0, upBound= 10)
    x2 = pl.LpVariable("x2",lowBound=0, upBound= 10)
    theta = [pl.LpVariable(f"theta_{k}", lowBound=-1e12, upBound=1e12) for k in range(K)]

    #create objective function
    prob += 7.0 * x1 + 11.0 * x2 + pl.lpSum(p_list[k] * theta[k] for k in range(K))

    #add constraints
    for k in range(K):
        for cut in cuts_k[k]:
            E_k, e_k = cut[0], cut[1]
            prob += E_k[0] * x1 + E_k[1] * x2 + theta[k] >= e_k

    #solve the model
    prob.solve(solver)
    # check the status
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError("Master not optimal:" + pl.LpStatus)

    #get the values
    x = np.array([pl.value(x1), pl.value(x2)], dtype= float)
    theta_val = np.array([pl.value(theta[k]) for k in range(K)])
    object = float(pl.value(prob.objective))

    return x, theta_val, object

#solve the second-stage dual problem
#dual problem: max pi^T b s.t. A^T pi <= q, pi >= 0
#A = [[1,2],[1,0]], b=[d1-x1, d2-x2]
def solve_dual(q1, q2, d1, d2, x, solver):
    x1, x2 = float(x[0]), float(x[1])

    #create problem
    prob = pl.LpProblem("dual", pl.LpMaximize)

    #create decision variable
    pi1 = pl.LpVariable("pi1", lowBound=0)
    pi2 = pl.LpVariable("pi2", lowBound=0)

    #create object
    prob += pi1 * (d1 -x1) + pi2 * (d2 - x2)

    #add constraints
    prob += pi1 + pi2 <= q1
    prob += 2 * pi1 + 0 * pi2 <= q2

    #solve the model
    prob.solve(solver)

    #check status
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError("Dual not optimal: " + pl.LpStatus[prob.status])

    #get value
    pi = np.array([pl.value(pi1), pl.value(pi2)], dtype= float)
    object_dual = float(pl.value(prob.objective))

    return pi, object_dual

#============main loop===================
def l_shaped_multi(start_x = None, max_iter = 100, tol =1e-6, msg = False):
    solver = get_solver(msg = msg)

    #Scenarios (p, q1, q2, d1, d2)
    scenarios = [
        (0.5, 26.0, 16.0, 6.0, 12.0),
        (0.5, 14.0, 24.0, 10.0, 4.0),
    ]

    # initilization
    K = len(scenarios)
    p_list = [scenarios[k][0] for k in range(K)]
    cuts_by_k = [[] for _ in  range(K)]

    if start_x is not None:
        x = np.array(start_x, dtype=float)
    else:
        x = np.zeros(2)

    for iter in range(1, max_iter+1):
        #step1 : solve the master problem
        x, theta_val, object = master_problem_multi(cuts_by_k, p_list, solver)
        print(f"\nIter {iter}: master x={x}, theta=[{', '.join(f'{t:.6f}' for t in theta_val)}], obj={object:.6f}")

        #check all the scenarios
        all_scenarios = True

        #Step2: skip

        #Step3: 3 solve the dual problem in different scenarios
        EQ = 0.0 #E[Q(x)]

        #for each scenario
        for k in range(K):
            p, q1, q2, d1, d2 = scenarios[k]
            pi_k, object_value_k = solve_dual(q1, q2, d1, d2, x, solver)
            EQ += p * object_value_k

            #calculate the cut value at current scenario w_k = pi_k ^ T (h_k - T_k x)
            # Here T_k = I2, h_k = [d1, d2]
            w_k = float(pi_k[0] * (d1 - x[0]) + pi_k[1] * (d2 - x[1]))

            #judge stop
            if theta_val[k] < w_k - tol:
                all_scenarios = False
                #add new cut
                E_k = np.array([pi_k[0], pi_k[1]], dtype= float)
                e_k = float(pi_k[0]*d1 + pi_k[1]*d2)
                cuts_by_k[k].append((E_k,e_k))
                print(f"  scen {k + 1}: add cut  {E_k[0]:.4f} x1 + {E_k[1]:.4f} x2 + theta_{k + 1} >= {e_k:.4f}")
            else:
                print(f"  scen {k + 1}: satisfied  (theta_k={theta_val[k]:.6f} >= w_k={w_k:.6f})")

        print(f"  E[Q(x)] = {EQ:.6f}")
        current_total_obj = 7.0 * x[0] + 11.0 * x[1] + EQ  # total_object = first_stage + second_stage
        print(f"  current_total_object = {current_total_obj:.6f}")
        if all_scenarios:
            print("Converged: all scenarios satisfied (multicut).")
            return x, theta_val, EQ, cuts_by_k

    raise RuntimeError("Not converged within max_iter")

if __name__ == "__main__":
    x_star, thetas, EQ_star, cuts_by_k = l_shaped_multi(start_x=(1.0, 5.0), msg=False)
    print("\n=== Final (multicut) ===")
    print("x*      =", x_star)
    print("theta_k =", thetas)
    print("E[Q]    =", EQ_star)
    total_obj_star = 7.0 * x_star[0] + 11.0 * x_star[1] + EQ_star
    print("optimal =", total_obj_star)
    for k, lst in enumerate(cuts_by_k, 1):
        for i, (E, e) in enumerate(lst, 1):
            print(f" scen{k} cut{i}: {E[0]:.4f} x1 + {E[1]:.4f} x2 + theta_{k} >= {e:.4f}")