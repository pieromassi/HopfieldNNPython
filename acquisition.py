import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image

# Fetch n-per_person <= 10 faces per person from the Olivetti dataset 
def fetch_and_select(n_per_person):

    # Sanity check
    if n_per_person < 1 or n_per_person > 10:
        print("Not a valid amount of faces")
        return

    # Fetches Olivetti faces
    data = fetch_olivetti_faces()

    # (person, image, height, width)
    images = data.images.reshape(40, 10, 64, 64)

    # Selects only the first n_per_person pictures for each person 
    selected = images[:, : n_per_person, :, :]

    # List of images as array
    return selected.reshape(40 * n_per_person, 64, 64) 


# Transforms a N x N image into a size x size image
# Then converts the image into a vector of +1 and -1
def reshape_and_binarize(img, size, threshold=None):

    # Converts img to pillow Image and applies bilinear interpolation
    reshaped_img = Image.fromarray(img).resize((size, size), resample=Image.BICUBIC) 

    # Back to array
    reshaped = np.asarray(reshaped_img)

    # Adaptive threshold
    if threshold is None:
        threshold = reshaped.mean()

    # If the pixel's value is above the threshold, put +1, else -1
    binarized = np.where(reshaped >= threshold, 1, -1).flatten()
    return binarized 


# Puts all together and returns a (P, N) array of +1 and -1
def load_patterns(n_per_person = 3, size = 32, threshold=None):

    # Get raw, selected images 
    raw_patterns = fetch_and_select(n_per_person)

    P = 40 * n_per_person

    # Iterate to get (P, N) array of reshaped, binarized, flattened patterns
    patterns = np.stack([reshape_and_binarize(raw_patterns[j, :, :], size) for j in range(P)])
    return patterns