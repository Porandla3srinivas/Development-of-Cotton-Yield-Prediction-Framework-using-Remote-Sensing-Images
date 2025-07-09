import numpy as np
import time


def mu_inv(y, mu):
    """
    This function generates a new point according to the lower and upper bounds
    and a random factor proportional to the current point.
    """
    return (((1 + mu) ** abs(y) - 1) / mu) * np.sign(y)


def bounds(s, lb, ub):
    """
    Apply bounds to the search agents to keep them within the search space.
    """
    s = np.maximum(s, lb)
    s = np.minimum(s, ub)
    return s


def MRA(Positions, fobj, lb, ub, T_max):
    """
    The Mud Ring Algorithm (MRA) main function.

    Parameters:
    - SearchAgents_no: Number of search agents
    - fobj: Objective function
    - lb: Lower bounds (can be a scalar or array)
    - ub: Upper bounds (can be a scalar or array)
    - T_max: Maximum number of iterations

    Returns:
    - MRLeader_score: Best score found
    - Convergence_curve: Convergence curve over iterations
    - MRLeader_pos: Position of the best agent found
    - time_taken: Total time taken by the algorithm
    """
    # Initialize the algorithm's parameters

    SearchAgents_no, dim = Positions.shape
    vLb = 0.6 * np.array(lb)
    vUb = 0.6 * np.array(ub)
    MRLeader_pos = np.zeros(dim)
    MRLeader_score = np.inf  # Set to -inf for maximization problems
    Boundary_no = len(ub)

    Convergence_curve = np.zeros(T_max)
    t = 0  # Loop counter
    start_time = time.time()
    v = np.random.rand(SearchAgents_no, dim)  # Velocity initialization

    # Main loop
    while t < T_max:
        for i in range(SearchAgents_no):
            # Enforce boundaries
            Positions[i, :] = bounds(Positions[i, :], lb[i], ub[i])

            # Calculate objective function value for each agent
            fitness = fobj(Positions[i, :])

            # Update the mud ring leader
            if fitness < MRLeader_score:
                MRLeader_score = fitness
                MRLeader_pos = Positions[i, :].copy()

        # Update the control parameter 'a' (Eq. 2 in the paper)
        a = 2 * (1 - t / T_max)

        # Update the positions of the search agents
        for i in range(SearchAgents_no):
            r = np.random.rand()  # Random number in [0,1]
            K = 2 * a * r - a  # Eq. (1) in the paper
            C = 2 * r  # Parameter in Eq. (4)
            l = np.random.rand()

            for j in range(dim):
                if abs(K) >= 1:
                    # Enforce velocity bounds and update positions (Eq. 3)
                    v[i, j] = bounds(v[i, j], vLb[i, j], vUb[i, j])
                    Positions[i, j] += v[i, j]
                else:
                    # Calculate and update positions (Eq. 4 and 5)
                    A = abs(C * MRLeader_pos[j] - Positions[i, j])
                    Positions[i, j] = mu_inv(bounds(MRLeader_pos[j] * np.sin(l * 2 * np.pi) - K * A, lb[i, j], ub[i, j]),
                                             np.random.rand())

        # Record convergence
        Convergence_curve[t] = MRLeader_score
        t += 1  # Increment loop counter

    # Calculate time taken
    time_taken = time.time() - start_time

    return MRLeader_score, Convergence_curve, MRLeader_pos, time_taken
