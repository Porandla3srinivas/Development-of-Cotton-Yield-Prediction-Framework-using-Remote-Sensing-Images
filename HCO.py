import numpy as np
import time


# The Hermit Crab Optimizer (HCO)
def HCO(X, obj_function, lb, ub, num_iterations):
    num_agents, dim = X.shape
    # X = np.random.uniform(lb, ub, (num_agents, dim))  # Initial positions of agents
    P = np.copy(X)  # Position memory for agents
    B = np.zeros(num_agents)  # Boolean indicator for direction
    Pi = obj_function(X)  # Objective values for positions in X
    f = np.inf * np.ones(num_agents)  # Fitness array
    A = np.zeros(num_agents)  # Indicator of failure (0 = success, 1 = failure)
    S = np.zeros((num_agents, dim))  # State matrix to record state
    convergence_curve = []  # To record best fitness over iterations
    ct = time.time()
    for iter in range(num_iterations):
        g = 0  # Total shells found
        b = 0  # Perceived risk of shell availability
        c = 0  # Indicator of distraction
        for i in range(num_agents):
            # Check if boundaries are violated
            X[i, :] = np.clip(X[i, :], lb[i], ub[i])
            # Evaluate the fitness
            Pi[i] = obj_function(X[i, :])
            if Pi[i] < f[i]:
                # Update the agent's personal best
                P[i, :] = X[i, :]
                f[i] = Pi[i]
                A[i] = 0  # Success
                S[i] = min(X[i, :])
            else:
                A[i] = 1  # Failure
                if b == 0:
                    B[i] = max(X[i, :])
                c += 1  # Indicator of distraction

        # Sorting agents by fitness
        sorted_indices = np.argsort(f)
        P = P[sorted_indices]
        f = f[sorted_indices]
        S = S[sorted_indices]

        # Record convergence data
        convergence_curve.append(np.min(f))

        # Post-processing
        b = b * (1 - (iter / num_iterations))
        if np.count_nonzero(f == 0) > 0:
            f[f == 0] = np.inf  # Avoid zero fitness in f

        # Update the position of agents for the next iteration
        for n in range(num_agents):
            if f[n] == 1:
                # Solitary search
                X[n, :] = P[n, :] + np.random.uniform(-1, 1, dim) * (B[n] - X[n, :])
            else:
                # Social search
                X[n, :] = S[n, :] + np.random.uniform(-1, 1, dim) * (P[(n + 1) % num_agents, :] - X[n, :])

    best_solution = S[0]
    best_fitness = f[0]
    ct = time.time() - ct
    return best_fitness, convergence_curve, best_solution, ct


if __name__ == '__main__':
    from numpy import matlib


    def objfun_cls(Soln):
        Fitn = np.zeros(Soln.shape[0])
        dimension = len(Soln.shape)
        if dimension == 2:
            for i in range(Soln.shape[0]):
                sol = np.round(Soln[i, :]).astype(np.int16)
                Fitn[i] = np.random.rand()
            return Fitn
        else:

            sol = np.round(Soln).astype(np.int16)
            Fitn = np.random.rand()
            return Fitn

    Npop = 10
    Chlen = 4  # hidden neuron count, Epoch in Mobilenet, hidden neuron count, Epoch in ResneXt
    xmin = matlib.repmat(np.asarray([5, 5, 5, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 255, 50]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("HCO...")
    [bestfit1, fitness1, bestsol1, time1] = HCO(initsol, fname, xmin, xmax, Max_iter)  # MAO
