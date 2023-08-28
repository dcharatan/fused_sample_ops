from typing import Tuple

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from . import _cuda

# The bounds' format is [[x_min, y_min], [x_max, y_max]].
Bounds = Float[Tensor, "2 2"]

# The format is (height, width).
ImageShape = Int64[Tensor, "2"]


def render_points(
    canvas: Float[Tensor, "sample 4"],
    samples: Float[Tensor, "sample 2"],
    points: Float[Tensor, "point 2"],
    colors: Float[Tensor, "point 3"],
    outer_radii: Float[Tensor, " point"],
    inner_radii: Float[Tensor, " point"],
    bounds: Bounds,
    image_shape: ImageShape,
):
    return _cuda.render_points(
        canvas, samples, points, colors, outer_radii, inner_radii, bounds, image_shape
    )


def get_point_rendering_function(
    points: Float[Tensor, "point 2"],
    colors: Float[Tensor, "point 3"],
    outer_radii: Float[Tensor, " point"],
    inner_radii: Float[Tensor, " point"],
    bounds: Bounds,
    image_shape: Tuple[int, int],
):
    def color_function(
        xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:
        p, _ = xy.shape
        canvas = torch.zeros((p, 4), dtype=xy.dtype, device=xy.device)
        render_points(
            canvas,
            xy,
            points,
            colors,
            outer_radii,
            inner_radii,
            bounds,
            torch.tensor(image_shape, dtype=torch.int64, device=xy.device),
        )
        return canvas

    return color_function
