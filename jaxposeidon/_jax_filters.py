"""JAX-pure 1D Gaussian filter.

Port of `scipy.ndimage.gaussian_filter1d(..., mode='nearest')` (the
edge-extending boundary handling). Used by `_instruments.py:59` and
`_high_res.py` for instrument PSF / convolution kernels.

scipy convention (`scipy/ndimage/_filters.py:_gaussian_kernel1d`):
- `radius = int(truncate * sigma + 0.5)` (`truncate` default `4.0`)
- weights = exp(-x^2 / (2 sigma^2)) for x in [-radius, radius],
  normalised to sum to 1
- boundary: `mode='nearest'` (value at index 0 / N-1 extends to the
  pad region — equivalent to "edge" padding).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def _kernel(sigma: float, truncate: float = 4.0):
    radius = int(truncate * float(sigma) + 0.5)
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float64)
    w = jnp.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
    return w / w.sum()


def gaussian_filter1d_edge(arr, sigma: float, truncate: float = 4.0):
    """1D Gaussian convolution with `mode='nearest'` (edge) boundary.

    Bit-equivalent to `scipy.ndimage.gaussian_filter1d(arr, sigma,
    mode='nearest', truncate=truncate)` for float64 input.

    `sigma` and `truncate` must be Python floats (kernel size is
    static; under jit, pass them via `static_argnames`).
    """
    arr = jnp.asarray(arr, dtype=jnp.float64)
    w = _kernel(sigma, truncate)
    radius = (w.shape[0] - 1) // 2
    if radius == 0:
        return arr
    n = arr.shape[0]
    padded = jnp.concatenate(
        [jnp.full((radius,), arr[0]), arr, jnp.full((radius,), arr[-1])]
    )
    k = w.shape[0]
    windows = jnp.stack([padded[i : i + n] for i in range(k)], axis=0)
    return jnp.einsum("i,ij->j", w, windows)
