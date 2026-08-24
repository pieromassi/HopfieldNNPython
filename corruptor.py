import numpy as np
from math import isqrt

# Flips each pixel independently with probability p
def add_noise(image, p, rng):
    flip = rng.random(image.shape[0]) < p
    image[flip] *= -1
    return image

# Flips a height x width region starting at (row, column) 
def crop(pattern, height, width, row, col):
    size = isqrt(pattern.shape[0])
    image = pattern.reshape(size,size)
    image[row:row + height, col:col + width] *= -1
    return image.flatten()

# Combines the two
def corruptor(image, mode, p=None, rng=None, height=None, width=None, row=None, col=None):
    rng = rng or np.random.default_rng()
    state = image.copy()

    if mode == "noise":
        return add_noise(state, p, rng)

    elif mode == "crop":
        size = isqrt(image.shape[0])
        if row is None:
            row = rng.integers(0, size - height + 1)
        if col is None:
            col = rng.integers(0, size - width + 1)
        return crop(state, height, width, row, col)

    else: 
        raise ValueError(f"Unknown corruption mode: {mode}")