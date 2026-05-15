"""JAX-pure interpolators for the v1 hot path.

- `pchip_interpolate(x, y, xq)`: monotone piecewise cubic Hermite
  interpolation (Fritsch-Carlson, 1980); parity to
  `scipy.interpolate.PchipInterpolator` / `scipy.interpolate.pchip_interpolate`.
  Used by `_atmosphere.py:230, :247`.
- `regular_grid_interp_linear(grid_axes, values, query_points)`:
  multidimensional linear interpolation on a rectilinear grid; boundary
  clip (no extrapolation). Used by `_chemistry.py:121`, `_clouds.py:162`.
"""

import jax.numpy as jnp


def _pchip_derivatives(x, y):
    h = x[1:] - x[:-1]
    delta = (y[1:] - y[:-1]) / h
    n = x.shape[0]

    d_inner_num = 3.0 * (h[:-1] + h[1:])
    d_inner_denom = (2.0 * h[1:] + h[:-1]) / delta[:-1] + (
        h[1:] + 2.0 * h[:-1]
    ) / delta[1:]
    d_inner = jnp.where(delta[:-1] * delta[1:] > 0.0, d_inner_num / d_inner_denom, 0.0)

    def _end_deriv(h0, h1, d0, d1):
        d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        d = jnp.where(jnp.sign(d) != jnp.sign(d0), 0.0, d)
        d = jnp.where(
            (jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(d) > 3.0 * jnp.abs(d0)),
            3.0 * d0,
            d,
        )
        return d

    d_first = _end_deriv(h[0], h[1], delta[0], delta[1])
    d_last = _end_deriv(h[-1], h[-2], delta[-1], delta[-2])

    if n == 2:
        d = jnp.array([delta[0], delta[0]], dtype=y.dtype)
    else:
        d = jnp.concatenate([jnp.array([d_first]), d_inner, jnp.array([d_last])])
    return d


def pchip_interpolate(x, y, xq):
    """Monotone PCHIP interpolation.

    `x`, `y`: 1D arrays of length N (knots; x strictly increasing).
    `xq`: query points (any shape).
    Returns array of shape `xq.shape`.

    Out-of-range behaviour matches scipy `PchipInterpolator`
    (`extrapolate=True` default): polynomial is evaluated on the
    nearest end interval.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    xq = jnp.asarray(xq, dtype=jnp.float64)
    n = x.shape[0]
    d = _pchip_derivatives(x, y)

    idx = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    d0 = d[idx]
    d1 = d[idx + 1]
    h = x1 - x0
    t = (xq - x0) / h

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1


def regular_grid_interp_linear(grid_axes, values, query_points):
    """Multidimensional linear interpolation on a rectilinear grid.

    `grid_axes`: tuple of N 1D arrays (axis grids; each strictly
    increasing).
    `values`: array of shape (len(grid_axes[0]), len(grid_axes[1]), ...).
    `query_points`: array of shape (..., N).

    Boundary: clipped (no extrapolation); points outside the grid take
    the boundary value.
    """
    values = jnp.asarray(values, dtype=jnp.float64)
    query_points = jnp.asarray(query_points, dtype=jnp.float64)
    ndim = len(grid_axes)

    batch_shape = query_points.shape[:-1]
    qp = query_points.reshape(-1, ndim)

    indices = []
    weights = []
    for d in range(ndim):
        axis = jnp.asarray(grid_axes[d], dtype=jnp.float64)
        m = axis.shape[0]
        q = qp[:, d]
        q_clip = jnp.clip(q, axis[0], axis[-1])
        i = jnp.clip(jnp.searchsorted(axis, q_clip, side="right") - 1, 0, m - 2)
        x0 = axis[i]
        x1 = axis[i + 1]
        w = (q_clip - x0) / (x1 - x0)
        indices.append(i)
        weights.append(w)

    out = jnp.zeros(qp.shape[0], dtype=jnp.float64)
    for corner in range(2**ndim):
        idx_list = []
        weight = jnp.ones(qp.shape[0], dtype=jnp.float64)
        for d in range(ndim):
            bit = (corner >> d) & 1
            idx_list.append(indices[d] + bit)
            weight = weight * (weights[d] if bit == 1 else (1.0 - weights[d]))
        out = out + weight * values[tuple(idx_list)]

    return out.reshape(batch_shape)
