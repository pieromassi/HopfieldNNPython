import numpy as np

def energy(state, weights):
    return -0.5 * state @ weights @ state

def convergence(pattern, weights, maxit=1000, rng=None):
    rng = rng or np.random.default_rng()

    state = pattern.copy()
    N = state.shape[0]

    E = energy(state,weights)
    energies = [E]

    for j in range(maxit):
        prev_state = state.copy()

        # Sweep over all neurons in random order
        for i in rng.permutation(N):

            # Weighted mean over neurons
            mean = weights[i] @ state

            s_old = state[i]
            s_new = 1 if mean >= 0 else -1

            # Update state and energy
            if s_new != s_old:
                E += 2 * s_old * mean
                state[i] = s_new

            energies.append(E)

        # Compare states after a sweep
        if np.array_equal(state, prev_state):
            break

    else:
        print(f"Warning: did not converge within {maxit} iterations")

    return state, np.array(energies)