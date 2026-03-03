from __future__ import annotations

import numpy as np


def smallest_angle_2d(v1, v2):
    """
    Smallest angle between v1 and v2 in radians, range (0, pi].
    """
    x1, y1 = v1
    v2 = v2.reshape(-1, 2)
    x2 = v2[:, 0]
    y2 = v2[:, 1]
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    return np.abs(np.arctan2(det, dot))


def signed_angle_2d(v1, v2):
    """
    Signed counterclockwise angle between v1 and v2 in radians, range (0, pi].
    """
    x1, y1 = v1
    v2 = v2.reshape(-1, 2)
    x2 = v2[:, 0]
    y2 = v2[:, 1]
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    return np.arctan2(det, dot)


def l2_norm(input_array: np.ndarray) -> np.ndarray:
    """
    Compute the l2 norm over the last dimension of an array.

        Parameters
        ----------
        input_array: (*, 2)

        Returns
        -------
        l2_norm
    """
    input_array = np.array(input_array, dtype=np.float32)
    l2_norm = np.sqrt(np.sum(input_array**2, axis=-1))
    return l2_norm
