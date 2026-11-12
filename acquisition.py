"""Load class-balanced image samples and convert them into ±1 patterns."""

import gzip
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces


DATASETS = {
    "olivetti": 64,
    "mnist": 28,
    "fashion-mnist": 28,
    "cifar-10": 32,
}

def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")


def _download(url, destination):
    """Cache completed downloads; discard partial downloads on failure."""
    destination = Path(destination)
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as out:
            temporary = Path(out.name)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, out)

        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return destination


def rgb_to_grayscale(image):
    """README conversion: grayscale = (R + G + B) / 3."""
    rgb = np.asarray(image, dtype=np.float32)

    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
        or not np.isfinite(rgb).all()
    ):
        raise ValueError("Expected a finite (height, width, 3) RGB image")

    # Convert before averaging to avoid uint8 arithmetic overflow.
    return rgb.mean(axis=-1)


def reshape_and_binarize(img, size, threshold=None, *, binary=False):
    """Resize grayscale [0, 1], then return a float64 ±1 vector."""
    _positive_integer(size, "size")
    gray = np.asarray(img, dtype=np.float32)

    if (
        gray.ndim != 2
        or gray.size == 0
        or not np.isfinite(gray).all()
        or gray.min() < 0
        or gray.max() > 1
    ):
        raise ValueError("Expected a nonempty grayscale image in [0, 1]")

    if threshold is not None and (
        not np.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be finite and between 0 and 1")

    if binary and not np.isin(gray, [0, 1]).all():
        raise ValueError("A binary image must contain only 0 and 1")

    method = (
        Image.Resampling.NEAREST
        if binary else Image.Resampling.BILINEAR
    )
    resized = np.asarray(
        Image.fromarray(gray).resize((size, size), resample=method)
    )

    if threshold is None:
        # A fixed cutoff also handles uniformly black/white images.
        cutoff = (
            0.5
            if binary or resized.min() == resized.max()
            else float(resized.mean())
        )
    else:
        cutoff = threshold

    return np.where(resized > cutoff, 1.0, -1.0).ravel()


def fetch_and_select(n_per_person, data_home="data"):
    """Preserve the existing Olivetti helper and selection order."""
    _positive_integer(n_per_person, "n_per_person")
    if n_per_person > 10:
        raise ValueError("Olivetti has only 10 images per person")

    data = fetch_olivetti_faces(
        data_home=str(Path(data_home) / "olivetti")
    )
    images = data.images.reshape(40, 10, 64, 64)
    return images[:, :n_per_person].reshape(-1, 64, 64)


def _load_idx(dataset, cache):
    """Read the original MNIST/Fashion-MNIST training files."""
    if dataset == "mnist":
        root = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    else:
        root = (
            "https://raw.githubusercontent.com/zalandoresearch/"
            "fashion-mnist/master/data/fashion/"
        )

    payloads = []
    for name in (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ):
        path = _download(root + name, cache / dataset / name)
        with gzip.open(path, "rb") as stream:
            payloads.append(stream.read())

    image_bytes, label_bytes = payloads
    header = np.frombuffer(image_bytes[:16], dtype=">u4")
    label_header = np.frombuffer(label_bytes[:8], dtype=">u4")

    if (
        header.size != 4
        or label_header.size != 2
        or tuple(header[[0, 2, 3]]) != (2051, 28, 28)
        or label_header[0] != 2049
        or header[1] != label_header[1]
    ):
        raise ValueError("Invalid or mismatched IDX headers")

    count = int(header[1])
    if (
        len(image_bytes) != 16 + count * 784
        or len(label_bytes) != 8 + count
    ):
        raise ValueError("Truncated IDX dataset")

    images = np.frombuffer(
        image_bytes, dtype=np.uint8, offset=16
    ).reshape(count, 28, 28)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    return images, labels


def _decode_cifar_records(records):
    # CIFAR stores R, G, B planes separately, not interleaved pixels.
    images = (
        records[:, 1:]
        .reshape(-1, 3, 32, 32)
        .transpose(0, 2, 3, 1)
    )
    return images, records[:, 0]


def _load_cifar(cache):
    path = _download(
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        cache / "cifar-10-binary.tar.gz",
    )
    names = {
        f"cifar-10-batches-bin/data_batch_{i}.bin"
        for i in range(1, 6)
    }

    batches = []
    with tarfile.open(path, "r:gz") as archive:
        # Read members directly: no extraction or pickle loading.
        for member in archive:
            if member.name in names and member.isfile():
                if member.size != 10000 * 3073:
                    raise ValueError("Invalid CIFAR-10 batch size")

                with archive.extractfile(member) as stream:
                    records = np.frombuffer(
                        stream.read(), dtype=np.uint8
                    ).reshape(10000, 3073)

                batches.append(records)
                names.remove(member.name)

    if names:
        raise ValueError("CIFAR-10 archive is missing training batches")

    return _decode_cifar_records(np.concatenate(batches))


def _select_per_class(images, labels, count, rng):
    indices = []

    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if candidates.size < count:
            raise ValueError(
                f"Class {label} has only {candidates.size} images"
            )
        indices.extend(
            rng.choice(candidates, size=count, replace=False)
        )

    return images[np.asarray(indices)]


def load_patterns(
    n_per_person=3,
    size=32,
    threshold=None,
    *,
    dataset="olivetti",
    n_per_class=None,
    data_home="data",
    random_state=0,
):
    """Return (P, size²) patterns.

    Existing Olivetti calls remain valid.
    size=None selects the dataset's native resolution.
    New datasets use their training split.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    count = n_per_person if n_per_class is None else n_per_class
    _positive_integer(count, "images per class")

    size = DATASETS[dataset] if size is None else size
    _positive_integer(size, "size")
    cache = Path(data_home)

    if dataset == "olivetti":
        images = fetch_and_select(count, data_home)
    else:
        if dataset in ("mnist", "fashion-mnist"):
            images, labels = _load_idx(dataset, cache)
        else:
            images, labels = _load_cifar(cache)

        images = _select_per_class(
            images, labels, count,
            np.random.default_rng(random_state),
        )

    if threshold is None and dataset in ("mnist", "fashion-mnist"):
        threshold = 0.5

    patterns = []
    for image in images:
        gray = (
            rgb_to_grayscale(image)
            if dataset == "cifar-10"
            else image.astype(np.float32)
        )

        # Olivetti and silhouettes already use [0, 1].
        if dataset in ("mnist", "fashion-mnist", "cifar-10"):
            gray = gray / 255.0

        patterns.append(
            reshape_and_binarize(gray, size, threshold)
        )

    return np.stack(patterns)