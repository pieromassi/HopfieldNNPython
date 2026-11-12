import numpy as np
import convergence_appendix as aconv

def basic_convergence(patterns, pattern, field="cubic", beta=None, maxit=1000, rng=None):
    rng = rng or np.random.default_rng()

    state = pattern.copy()
    N = state.shape[0]

    # Compute overlaps with self interaction
    overlaps = (patterns @ state) / N

    energies = []

    for j in range(maxit):
        prev_state = state.copy()

        # Sweep over all neurons in random order
        for i in rng.permutation(N):

            overlaps_iter = overlaps - patterns[:,i] * state[i] / N

            # Weighted mean over neurons
            mean = aconv.weighted_mean(patterns, overlaps_iter, i, field=field, beta=beta)

            s_old = state[i]
            s_new = 1 if mean >= 0 else -1

            # Update state and energy
            if s_new != s_old:
                overlaps += patterns[:, i] * (s_new - s_old) / N 
                state[i] = s_new
                energies.append(aconv.energy(patterns, overlaps_iter, field=field, beta=beta))

        # Compare states after a sweep
        if np.array_equal(state, prev_state):
            break

    else:
        print(f"Warning: did not converge within {maxit} iterations")

    return state, np.array(energies)


def deterministic(patterns, pattern, field="cubic", beta=None, maxit=1000, rng=None):
    rng = rng or np.random.default_rng()

    state = pattern.copy()
    N = state.shape[0]

    # Compute overlaps with self interaction
    overlaps = (patterns @ state) / N

    # Energy initialization
    current_energy = aconv.energy(patterns, overlaps, field=field, beta=beta)
    energies = [current_energy]

    converged = False

    for sweep in range(maxit):
        prev_state = state.copy()

        accepted = 0

        # Sweep over all neurons in random order
        for i in rng.permutation(N):
            s_old = state[i]

            # Weighted mean over neurons
            mean = aconv.weighted_mean(patterns, overlaps, i, field=field, beta=beta)

            s_new = 1 if mean >= 0 else -1

            # Update state and energy
            if s_new != s_old:
                overlaps += patterns[:, i] * (s_new - s_old) / N 
                state[i] = s_new
                energies.append(aconv.energy(patterns, overlaps, field=field, beta=beta))
    
        # Compare states after a sweep
        if np.array_equal(state, prev_state):
            converged = True
            break

    if not converged:
        print(f"Warning: state not stable after {sweep + 1} sweeps")

    return state, np.array(energies)


def metropolis_hastings(patterns, pattern, field="cubic", beta=None, maxit=1000, rng=None):
    rng = rng or np.random.default_rng()

    state = pattern.copy()
    N = state.shape[0]

    # Compute overlaps with self interaction
    overlaps = (patterns @ state) / N

    # Energy initialization
    current_energy = aconv.energy(patterns, overlaps, field=field, beta=beta)
    energies = [current_energy]

    # Temperature estimation
    target_rate = 0.2
    uphill_energy_diff = aconv.sample_dE_uphill(patterns, state, overlaps, field=field, beta=beta)
    median_dE = np.median(uphill_energy_diff)
    T0 = aconv.Glauber_temp(median_dE, target_rate)
    T = aconv.mean_temp(uphill_energy_diff, T0, target_rate)

    converged = False

    for sweep in range(maxit):
        prev_state = state.copy()

        accepted = 0

        # Sweep over all neurons in random order
        for i in rng.permutation(N):
            s_old = state[i]

            overlaps_flip = overlaps - 2 * patterns[:,i] * s_old / N
            energy_flip = aconv.energy(patterns, overlaps_flip, field=field, beta=beta)
            dE = energy_flip - current_energy

            # Stochastic flip and updates (Glauber rule)
            p = aconv.accept_prob(dE, T)
            if rng.random() < p:
                state[i] *= -1
                overlaps = overlaps_flip
                current_energy = energy_flip
                energies.append(energy_flip)
    
        # Compare states after a sweep
        if np.array_equal(state, prev_state):
            converged = True
            break

    if not converged:
        print(f"Warning: state not stable after {sweep + 1} sweeps")

    return state, np.array(energies)


def simulated_annealing(patterns, pattern, field="cubic", beta=None, maxit=1000, rng=None):
    rng = rng or np.random.default_rng()

    state = pattern.copy()
    N = state.shape[0]

    # Compute overlaps with self interaction
    overlaps = (patterns @ state) / N

    # Energy initialization
    current_energy = aconv.energy(patterns, overlaps, field=field, beta=beta)
    energies = [current_energy]

    target_rate = 0.44     # empirical middle ground
    alpha_fast  = 0.90     # still hot -> cool aggressively
    alpha_slow  = 0.99     # near frozen -> cool gently
    frozen_streak = 0      # consecutive freezing count 

    # Temperature initialization
    uphill_energy_diff = aconv.sample_dE_uphill(patterns, state, overlaps, field=field, beta=beta)
    median_dE = np.median(uphill_energy_diff)
    T0 = aconv.Glauber_temp(median_dE, target_rate)
    T = aconv.mean_temp(uphill_energy_diff, T0, target_rate)

    converged = False

    for sweep in range(maxit):
        prev_state = state.copy()

        accepted = 0

        # Sweep over all neurons in random order
        for i in rng.permutation(N):
            s_old = state[i]

            overlaps_flip = overlaps - 2 * patterns[:,i] * s_old / N
            energy_flip = aconv.energy(patterns, overlaps_flip, field=field, beta=beta)
            dE = energy_flip - current_energy

            # Stochastic flip and updates (Glauber rule)
            p = aconv.accept_prob(dE, T)
            if rng.random() < p: 
                state[i] *= -1
                overlaps = overlaps_flip
                current_energy = energy_flip
                energies.append(energy_flip)
                accepted += 1              

        # Temperature update
        rate = accepted / N
        T *= alpha_fast if rate > target_rate else alpha_slow

        # Freezing criterion
        rate_perc = rate * 100
        if rate_perc < 0.05:
            frozen_streak += 1
            if frozen_streak == 50:
                break
        else:
            frozen_streak = 0
    
        # Compare states after a sweep
        if np.array_equal(state, prev_state):
            converged = True
            break

    if not converged:
        print(f"Warning: state not stable after {sweep + 1} sweeps")

    return state, np.array(energies)