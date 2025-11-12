"""
# p196 exercise2
# min 7 * x1 + 11 * x2 + E[q1 y1 + q2 y2]
# s.t.  y1 + 2 * y2 >= d1 - x1
#       y1        >= d2 - x2
#       0 <= x1, x2 <= 10,  y >= 0
# (q1,q2,d1,d2) = (26,16,6,12) p = 0.5,  (14,24,10,4) p = 0.5
"""

import numpy as np
import pulp as pl

#choose solver
def get_solver(msg = False):
    try:
        return pl.GUROBI_CMD(msg = msg)
    except Exception:
        return pl.PULP_CBC_CMD(msg = msg)

#solve the linear problem
def master_problem(cuts, solver):
    #create problem
    prob = pl.LpProblem("master_prob", pl.LpMinimize)

    #create decision variable
    x1 = pl.LpVariable("x1", lowBound=0, upBound=10)
    x2 = pl.LpVariable("x2", lowBound=0, upBound=10)
    theta = pl.LpVariable("theta", lowBound=-1e12, upBound=1e12)

    #create object function
    prob += 7.0 * x1 + 11 * x2 + theta

    #adding cutting constrains
    for cut in cuts:
        E = cut[0]
        e = cut[1]
        prob += E[0] * x1 + E[1] * x2 + theta >= e #E_lx + theta >= e_l

    #solve the model
    prob.solve(solver)

    #check the status
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError("Master not optimal:" + pl.LpStatus[prob.status])

    #get the values
    x = np.array([pl.value(x1),pl.value(x2)], dtype= float)
    theta_optimal = float(pl.value(theta))
    object = float(pl.value(prob.objective))

    return x, theta_optimal, object

#solve the second-stage dual problem
#dual problem: max pi^T b s.t. A^T pi <= q, pi >= 0
#A = [[1,2],[1,0]], b=[d1-x1, d2-x2]
def solve_dual(q1, q2, d1, d2, x, solver):
    x1, x2 = float(x[0]), float(x[1])
    #create problem
    prob = pl.LpProblem("dual", pl.LpMaximize)
    pi1 = pl.LpVariable("pi1", lowBound=0)
    pi2 = pl.LpVariable("pi2", lowBound=0)

    #create object function
    prob += pi1 * (d1 - x1) + pi2 * (d2 - x2)

    #create constraint(A^T pi <= 1)
    prob += (pi1 + pi2) <= q1
    prob += (2 * pi1 + 0 * pi2) <= q2

    #solve the dual problem
    prob.solve(solver)

    #check the status
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError("Dual not optimal: " + pl.LpStatus[prob.status])

    #get values
    pi = np.array([pl.value(pi1), pl.value(pi2)], dtype= float)
    object_dual = pl.value(prob.objective)

    return pi, object_dual

#==================main loop=========================================
#L-shape loop
def l_shape(start_x = None, max_iter = 100, tol = 1e-6, msg =False):
    solver = get_solver(msg =msg)

    #different scenario (p, q1, q2, d1, d2)
    scenarios = [(0.5, 26.0, 16.0, 6.0, 12.0),
                 (0.5, 14.0, 24.0, 10.0, 4.0)]

    #initilization
    cuts = []
    if start_x is not None:
        x = np.array(start_x, dtype=float)
    else:
        x = np.zeros(2)

    for iter in range(1, max_iter + 1):
        #Step 1: solve the linear master problem
        x, theta_optimal, object = master_problem(cuts, solver)
        print(f"\nIter {iter}: master x={x}, theta={theta_optimal:.6f}, obj={object:.6f}")

        #step 2 can be skipped

        #Step 3 solve the dual problem in different scenarios
        #initilize the E and e
        E = np.zeros(2)  # Σ p_k (pi_k^T T_k)
        e = 0.0          # Σ p_k (pi_k^T h_k)
        EQ = 0.0

        for (p,q1, q2, d1, d2) in scenarios:
            pi, object_dual = solve_dual(q1, q2, d1, d2, x, solver)
            EQ += p * object_dual
            E += p * pi
            e += p * (pi[0]*d1 + pi[1]*d2)
            print(f"  scen q=({q1},{q2}), d=({d1},{d2}) | pi={pi}, object_dual={object_dual:.6f}")

        #calculate the cut at x w = e - E x
        w = float(e - E.dot(x))
        print(f"  E[Q(x)]={EQ:.6f}, cut_at_x={w:.6f}")

        #stop check
        if theta_optimal >= w - tol:
            print("Converged: theta >= cut value at x")
            return x, theta_optimal, EQ, cuts
        else:
            #add a new cut
            cuts.append((E.copy(), float(e))) #(cut_parameter, Constant term)

    raise RuntimeError("Not converged within max_iter")

#==================================
if __name__ == "__main__":
    x_star, theta_star, EQ_star, cuts = l_shape(start_x=(1.0, 5.0), msg=False)
    print("\n=== Final ===")
    print("x*     =", x_star)
    print("theta* =", theta_star, "  (≈ Q(x*))")
    print("E[Q]   =", EQ_star)
    print("#cuts  =", len(cuts))
    for i, (E, e) in enumerate(cuts, 1):
        print(f"cut {i}: {E[0]:.4f} x1 + {E[1]:.4f} x2 + theta >= {e:.4f}")