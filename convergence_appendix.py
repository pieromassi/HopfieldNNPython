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


# Numerically stable sigmoid
# Source - https://stackoverflow.com/a/64717799
# Posted by Szymon Maszke, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-29, License - CC BY-SA 4.0

def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocat
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result


# Acceptance probability
def accept_prob(dE, T):
    x = -dE / T
    return sigmoid(x)


# Sample energy differences
def sample_dE_uphill(patterns, state, overlaps, field="cubic", beta=None):
    N = state.shape[0]

    flip_energies = []
    E = energy(patterns, overlaps, field, beta)

    for i in range(N):
        overlaps_flip = overlaps - 2 * patterns[:,i] * state[i] / N
        energy_flip = energy(patterns, overlaps_flip, field=field, beta=beta)
        flip_energies.append(energy_flip)

    energy_diff = np.array(flip_energies) - E
    return energy_diff[energy_diff > 0]

# Compute median temperature
def Glauber_temp(dE, p0):
    return dE / np.log((1/p0) - 1)

# Converge to mean temperature
# Geometric bisection with mean Glauber probabilities
def mean_temp(uphill_energy_diff, T0, p0, tol=1e-6, maxit=1000):
    mid = T0 # initial guess
    low, high = mid * 1e-3, mid * 1e3
    
    for i in range(maxit):
        mid = np.sqrt(low * high)
        rate = sigmoid(-uphill_energy_diff / mid).mean()
        low, high = (mid, high) if rate < p0 else (low, mid)

        if np.abs(rate - p0) / p0 < tol:
            break
    else:
        print(f"Warning: T calibration did not converge within {maxit} iterations "
          f"(bracket [{T0*1e-3:.3g}, {T0*1e3:.3g}] may not contain the target)")

    return mid