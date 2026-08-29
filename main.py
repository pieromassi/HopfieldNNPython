import os
import numpy as np
import matplotlib.pyplot as plt
import acquisition as a
import learning as learn
import convergence as conv
import corruptor as cor
import ask_user as ask
from math import isqrt

def main():

    patterns = a.load_patterns(n_per_person=10, size=64, threshold=None)
    P, N = patterns.shape
    corr = (patterns @ patterns.T) / N
    off_diag = corr[~np.eye(P, dtype=bool)]
    print("mean overlap:", off_diag.mean(), "max:", off_diag.max())
    l = len(patterns)

    # # Check if there is weights.npy, otherwise creates it
    # if os.path.exists("weights.npy"):
    #     weights = np.load("weights.npy")
    # else:
    #     weights = learn.learn(patterns)
        
    #     # Saves the weights array in a new .npy file
    #     np.save("weights", weights)

    # Choose an image to corrupt and try to reconstruct
    while True:
        try:
            idx = int(input(f"Choose an image (1 - {l}): ")) - 1
            if not (0 <= idx < l):
                raise IndexError
            pattern = patterns[idx]
            break
        except (ValueError, IndexError):
            print("Choose a valid image")

    size = isqrt(pattern.shape[0])

    # Corrupts the image as the user wants to
    mode = ask.ask_mode()

    if mode == "noise":
        p = ask.ask_noise()
        corrupted = cor.corruptor(pattern, mode, p)
    else: 
        height, width = ask.ask_crop(size)
        corrupted = cor.corruptor(pattern, mode, height=height, width=width)

    field = ask.ask_field()
    self_int = ask.ask_self_int()
    beta = ask.ask_beta() if field == "exp" else None

    result, energies = conv.convergence(patterns, corrupted, field=field, self_int=self_int, beta=beta)

    # # Save result in .npy file
    # if not os.path.exists("stranger"):
    #     np.save("stranger", state)

    # Display the original, corrupted and reconstructed image, side by side
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    images = [pattern, corrupted, result]
    titles = ["Original", "Corrupted", "Reconstructed"]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image.reshape(size,size), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()

    plt.show()

    # Energy plot, should be decreasing
    plt.figure(figsize=(6, 4))
    plt.plot(energies, color="tab:blue")
    plt.title(f"Energy over neuron updates: {field} field")
    plt.xlabel("Updates")
    plt.ylabel("Energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': main()