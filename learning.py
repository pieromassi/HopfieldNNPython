import numpy as np
import acquisition as a

# Computes the NxN array of weights by Hebbs' rule
def learn(patterns):
    N = patterns.shape[1]

    # Enables BLAS matmul
    patterns = patterns.astype(np.float32)

    # Hebbs' rule with wrong diagonal
    # @ = np.matmul, .T = transpose
    weights = (patterns.T @ patterns) / N

    # In-place, avoids building an NxN identity matrix
    np.fill_diagonal(weights, 0)        

    return weights


if __name__ == "__main__":

    # Loads (P, N) = (120, 32 * 32) array of patterns
    patterns = a.load_patterns(n_per_person=3, size=32, threshold=None)

    weights = learn(patterns)

    # Saves the weights array in a .npy file
    np.save("weights", weights)