import random
import time
import numpy as np
import random as rn


def getPheromone(fit, min, max):  # Eq.12 in the paper
    o = np.zeros((fit.shape[0]))
    for i in range(fit.shape[0]):
        o[i] = (max - fit[i]) / (max - min)
    return o


def getBinary():  # algorithm-2 in the paper page 6.
    if np.random.random() < 0.5:
        val = 0
    else:
        val = 1
    return val


def BWO(Positions, fobj, Lb, Ub, Max_iter):
    ### Black Widow Algorithm
    N,dim = Positions.shape[0], Positions.shape[1]
    # ub = Ub[0, :]
    # lb = Lb[1, :]
    Fitness = np.zeros((N))
    for i in range(N):
        Fitness[i] = fobj(Positions[i, :])

    vMin = min(Fitness)  # the min fitness value vMin and the position minIdx
    theBestVct = Positions[0, :]  # the best vector
    vMax = max(Fitness)  # the min fitness value vMin and the position minIdx
    Convergence_curve = np.zeros((1, Max_iter))
    Convergence_curve = np.zeros((Max_iter))
    Convergence_curve[0] = vMin
    pheromone = getPheromone(Fitness, vMin, vMax)
    ct = time.time()
    t = -1  # Loop counter
    # Main loop
    for t in range(1, Max_iter):
        beta = -1 + 2 * np.random.random()  # -1 < beta2 < 1     section 3.2.1 in the paper, page 4
        m = 0.4 + 0.5 * np.random.random()  # 0.4 < m < 0.9
        v = np.zeros((N, dim))
        for r in range(N):
            P = np.random.random()
            r1 = round(1 + (N - 1) * np.random.random())
            if P >= 0.3:  # spiral search   Eq. 11 in the paper, page 4
                v[r, :] = theBestVct - np.cos(2 * np.pi * beta) * Positions[r, :]
            else:  # direct search Eq. 11
                v[r, :] = theBestVct - m * Positions[r1, :]

            if pheromone[r] <= 0.3:
                band = 1
                while band:
                    r1 = round(1 + (N - 1) * np.random.random())
                    r2 = round(1 + (N - 1) * np.random.random())
                    if r1 != r2:
                        band = 0
                        # pheromone function.  Eq. 13 page 5 , getBinary is the  algorithm-2 in the paper, page 6
                v[r, :] = theBestVct + (Positions[r1, :] - ((-1) ^ getBinary()) * Positions[r2, :]) / 2;

            # ********************************************************************************
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = v[r, :] > Ub
            Flag4lb = v[r, :] < Lb
            v[r, :] = (v[r, :] * (~(Flag4ub + Flag4lb))) + Ub * Flag4ub + Lb * Flag4lb
            # Evaluate new solutions
            Fnew = fobj(v[r, :])
            # Update if the solution improves
            if Fnew <= Fitness[r]:
                Positions[r, :] = v[r, :]
                Fitness[r] = Fnew

            if Fnew <= vMin:
                theBestVct = v[r, :]
                vMin = Fnew

        # update max and pheromons
        [vMax, maxIdx] = max(Fitness)
        pheromone = getPheromone(Fitness, vMin, vMax)
        Convergence_curve[t + 1] = vMin
    ct = time.time() - ct
    # **********************************[End  BWOA function]

    return vMin, Convergence_curve, theBestVct, ct
