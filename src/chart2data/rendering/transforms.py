"""Coordinate conversion helpers using persisted homogeneous transforms."""

from __future__ import annotations

import numpy as np

from chart2data.schema import TransformMetadata


def _apply(matrix: list[list[float]] | np.ndarray, points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.shape[-1] != 2:
        raise ValueError("points must end in an (x, y) dimension")
    flat = array.reshape(-1, 2)
    homogeneous = np.column_stack([flat, np.ones(len(flat))])
    transformed = homogeneous @ np.asarray(matrix, dtype=float).T
    return transformed[:, :2].reshape(array.shape)


def data_to_pixel(points: np.ndarray, transform: TransformMetadata) -> np.ndarray:
    return _apply(transform.data_to_pixel_matrix, points)


def pixel_to_data(points: np.ndarray, transform: TransformMetadata) -> np.ndarray:
    return _apply(transform.pixel_to_data_matrix, points)


def matrices_from_matplotlib(display_matrix: np.ndarray, image_height: int) -> TransformMetadata:
    """Convert Matplotlib's bottom-left display coordinates to PNG top-left pixels."""
    flip_y = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(image_height)], [0.0, 0.0, 1.0]])
    forward = flip_y @ np.asarray(display_matrix, dtype=float)
    inverse = np.linalg.inv(forward)
    return TransformMetadata(
        data_to_pixel_matrix=forward.tolist(), pixel_to_data_matrix=inverse.tolist()
    )


def matrices_from_affine(
    scale_x: float, offset_x: float, scale_y: float, offset_y: float
) -> TransformMetadata:
    """Build top-left-origin affine transforms of the form
    pixel_x = scale_x * data_x + offset_x and pixel_y = scale_y * data_y + offset_y."""
    if scale_x == 0 or scale_y == 0:
        raise ValueError("affine scale factors must be non-zero")
    forward = np.array(
        [
            [scale_x, 0.0, offset_x],
            [0.0, scale_y, offset_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    inverse = np.linalg.inv(forward)
    return TransformMetadata(
        data_to_pixel_matrix=forward.tolist(), pixel_to_data_matrix=inverse.tolist()
    )
