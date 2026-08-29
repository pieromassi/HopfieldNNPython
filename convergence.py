import numpy as np

# Energy at current state
def energy(patterns, overlaps, field="cubic", beta=None):
    N = patterns.shape[1]

    if field == "hebb":
        F = (overlaps ** 2) / 2

    elif field == "cubic":
        F = (overlaps ** 4) / 4

    elif field == "exp":
        if beta is None:
            raise ValueError("beta must be specified for the exponential field.")
        F = np.exp(beta * overlaps) / beta

    else:
        raise ValueError(f"Unknown field: {field}")

    return -N * F.sum()


# Weighted mean over neurons
def weighted_mean(patterns, overlaps, idx, field="cubic", beta=None):

    if field == "hebb":
        f = overlaps

    elif field == "cubic":
        f = overlaps ** 3

    elif field == "exp":
        if beta is None:
            raise ValueError("beta must be specified for the exponential field.")
        f = np.exp(beta * overlaps) 

    else:
        raise ValueError(f"Unknown field: {field}")

    return f @ patterns[:, idx]


def convergence(patterns, pattern, field="cubic", self_int=True, beta=None, maxit=1000, rng=None):
    rng = rng or np.random.default_rng()

    state = pattern.copy()
    N = state.shape[0]

    # Compute overlaps with self interaction
    overlaps = (patterns @ state) / N

    energies = [0]

    for j in range(maxit):
        prev_state = state.copy()

        # Sweep over all neurons in random order
        for i in rng.permutation(N):

            overlaps_iter = overlaps if self_int else overlaps - patterns[:,i] * state[i] / N

            # Weighted mean over neurons
            mean = weighted_mean(patterns, overlaps_iter, i, field=field, beta=beta)

            s_old = state[i]
            s_new = 1 if mean >= 0 else -1

            # Update state and energy
            if s_new != s_old:
                overlaps += patterns[:, i] * (s_new - s_old) / N 
                state[i] = s_new
                energies.append(energy(patterns, overlaps_iter, field=field, beta=beta))

        # Compare states after a sweep
        if np.array_equal(state, prev_state):
            break

    else:
        print(f"Warning: did not converge within {maxit} iterations")

    return state, np.array(energies)