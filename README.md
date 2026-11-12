# Hopfield Neural Network

A Python implementation of associative memory for images. The network stores binary images and tries to retrieve one from a corrupted copy. It updates one pixel at a time and lets you compare classical Hebbian recall, nonlinear fields, and stochastic updates on the same task.

## Quick start

From the project directory:

```bash
python -m pip install numpy pillow scikit-learn matplotlib
python main.py
```

1. Choose a dataset and the index of a stored image. The first run may download the data into `data/`.
2. Choose `noise` to flip each pixel independently with the percentage you enter, or `crop` to invert a randomly placed rectangle of the requested height and width.
3. Choose a field: `hebb`, `cubic`, or `exp`. For `exp`, enter a finite `beta > 1`.
4. Choose whether to include self-interaction. Enter `no` for the update field that excludes the current pixel, or `yes` to choose among the full-overlap deterministic and stochastic modes.

For a first run, try **Olivetti → image 1 → noise → 15 → exp → 50 → no**. The program displays the original binary image, the corrupted input, the final state, and plots of the recorded energy and its differences. Close each plot window to advance if your Matplotlib backend blocks.

## Datasets

| Menu | Dataset | Images stored by default | Native size | Binarization |
| --- | --- | ---: | ---: | --- |
| 1 | [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html) | 120, first 3 per person across 40 people | 64 × 64 | Per-image mean |
| 2 | [MNIST](https://github.com/cvdfoundation/mnist) handwritten digits | 30, 3 per digit from the training split | 28 × 28 | Fixed threshold of 0.5 |
| 3 | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) clothing images | 30, 3 per class from the training split | 28 × 28 | Fixed threshold of 0.5 |
| 4 | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) color images | 30, 3 per class from the training split | 32 × 32 | Mean of RGB channels to grayscale, then per-image mean |

Images are normalized to grayscale values in `[0, 1]`, resized by bilinear interpolation if a different size is requested, and flattened into vectors with entries `+1` above the threshold and `−1` otherwise. Uniform images use a cutoff of 0.5 when the per-image mean would otherwise apply.

The CLI uses native resolution and a fixed seed for sampling the three labeled training datasets. To change the memory size, resolution, or threshold in Python, use `acquisition.load_patterns` with `n_per_class`, `size`, or `threshold`. Each run tests recall of a stored image from a corrupted copy. Correlated images can lead to incorrect or spurious recalls.

## Overlaps, fields, and energy

For each stored image $\xi^\mu$, the overlap with the current state $s$ is the normalized dot product

```math
m^\mu(s) = \frac{1}{N}\sum_{i=1}^N \xi_i^\mu s_i,
\qquad
E(s) = -N\sum_\mu F(m^\mu(s)).
```

An overlap of $1$ means an exact match, and $-1$ means every pixel is inverted. The deterministic field at pixel $i$ is the sum of $f(m^\mu)\xi_i^\mu$ over stored images. Its sign determines the next pixel value, with ties set to $+1$.

| Field | Update weight $f(m)$ | Energy term $F(m)$ | Effect |
| --- | --- | --- | --- |
| `hebb` | $m$ | $m^2/2$ | Linear weighting. Excluding the current pixel gives the classical zero-diagonal Hebbian rule. |
| `cubic` | $m^3$ | $m^4/4$ | Emphasizes large absolute overlaps while preserving their sign. |
| `exp` | $e^{\beta m}$ | $e^{\beta m}/\beta$ | Favors the largest positive overlaps. Increasing $\beta$ sharpens the weighting. |

The recall code keeps the stored patterns and updates their overlaps after a flip, so it uses no dense weight matrix. The separate `learning.py` script can build and save the classical Hebbian matrix.

## Recall dynamics

A sweep visits every pixel once in a fresh random order. With self-interaction disabled, the deterministic field omits the current pixel from each overlap. With self-interaction enabled, the `deterministic` option uses the full overlaps in the field.

**Fixed-temperature stochastic recall.** Select `metropolis-hastings` in the CLI. The implementation uses the Barker (Glauber) acceptance rule. A proposed flip of energy change $\Delta E=E_{\mathrm{flipped}}-E_{\mathrm{current}}$ is accepted with probability

```math
p_{\mathrm{flip}} = \frac{1}{1 + \exp(\Delta E/T)}.
```

At positive temperature, even uphill flips can occur and help the network escape a local minimum. Classical Metropolis acceptance for the same flip proposal is $\min(1,e^{-\Delta E/T})$. The program estimates $T$ from uphill moves at the starting state.

**Simulated annealing.** Select `simulated annealing` to use the same flip probability with a temperature that decreases after each sweep. Cooling is faster while many flips are accepted and slower as acceptance drops. This encourages exploration early and favors lower-energy moves later. Energy can rise during either stochastic mode. The current implementation can stop after a single sweep with no accepted flips.

## Use from Python

```python
import numpy as np

from acquisition import load_patterns
from convergence import basic_convergence
from corruptor import corruptor

patterns = load_patterns(dataset="mnist", n_per_class=3, size=None)
rng = np.random.default_rng(42)
cue = corruptor(patterns[0], "noise", p=15, rng=rng)
recalled, energies = basic_convergence(
    patterns, cue, field="exp", beta=50, maxit=100, rng=rng
)
```

`maxit` counts sweeps. The function returns a flat image vector and an energy trace. In the current self-interaction-free mode, the trace uses the overlap with the updated pixel omitted, so use the full-state energy for comparisons across modes.

## Observations

On Olivetti Faces, the most encouraging recalls so far used the exponential field with $\beta > 10$, usually $\beta = 50$. The other datasets have had limited testing.

## References

- J. J. Hopfield, [Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/doi/10.1073/pnas.79.8.2554) (1982).
- D. Krotov and J. J. Hopfield, [Dense Associative Memory for Pattern Recognition](https://arxiv.org/abs/1606.01164) (2016).
