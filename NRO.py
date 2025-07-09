import time

import numpy as np


def fission_operation(particle, num_fission_particles, lb, ub):
    fission_particles = []
    for _ in range(num_fission_particles):
        mutation_vector = np.random.uniform(lb, ub, particle.shape)
        fission_particles.append(particle + mutation_vector)
    return np.array(fission_particles)


# Fusion operation
def fusion_operation(particles):
    fused_particle = np.mean(particles, axis=0)
    return fused_particle


# Nuclear Reaction Optimization algorithm
def NRO(population, objective_function, VRmin, VRmax, num_iterations):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    num_fission_particles = 3

    best_particle = np.zeros((1, dim))
    best_fitness = float('inf')

    Convergence_curve = np.zeros((num_iterations, 1))

    t = 0
    ct = time.time()
    # Optimization loop
    for iteration in range(num_iterations):
        new_population = []

        for particle in population:
            # Fission operation
            fission_particles = fission_operation(particle, num_fission_particles, lb, ub)
            fission_fitness = np.array([objective_function(fp) for fp in fission_particles])

            # Select the best fission particle
            best_fission_particle = fission_particles[np.argmin(fission_fitness)]
            new_population.append(best_fission_particle)

            # Update the best particle found
            best_fission_fitness = np.min(fission_fitness)
            if best_fission_fitness < best_fitness:
                best_fitness = best_fission_fitness
                best_particle = best_fission_particle

        # Fusion operation
        if iteration % 10 == 0:
            fused_particle = fusion_operation(new_population)
            fused_fitness = objective_function(fused_particle)
            if fused_fitness < best_fitness:
                best_fitness = fused_fitness
                best_particle = fused_particle
            new_population.append(fused_particle)

        # Update population
        population = np.array(new_population)
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[num_iterations - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_particle, ct
