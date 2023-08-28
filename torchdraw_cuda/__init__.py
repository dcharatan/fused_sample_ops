from typing import Literal, Tuple

import torch
from jaxtyping import Float, Int32, Int64
from torch import Tensor

from . import _cuda

# The bounds' format is [[x_min, y_min], [x_max, y_max]].
Bounds = Float[Tensor, "2 2"]

# The format is (height, width).
ImageShape = Int64[Tensor, "2"]

CAP_TYPES = {
    "butt": 0,
    "round": 1,
    "square": 2,
}


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


def render_lines(
    canvas: Float[Tensor, "sample 4"],
    samples: Float[Tensor, "sample 2"],
    starts: Float[Tensor, "line 2"],
    ends: Float[Tensor, "line 2"],
    colors: Float[Tensor, "line 3"],
    widths: Float[Tensor, " line"],
    caps: Int32[Tensor, " line"],
    bounds: Bounds,
    image_shape: ImageShape,
):
    return _cuda.render_lines(
        canvas, samples, starts, ends, colors, widths, caps, bounds, image_shape
    )


def get_line_rendering_function(
    starts: Float[Tensor, "line 2"],
    ends: Float[Tensor, "line 2"],
    colors: Float[Tensor, "line 3"],
    widths: Float[Tensor, " line"],
    cap: Literal["butt", "round", "square"],
    bounds: Bounds,
    image_shape: Tuple[int, int],
):
    def color_function(
        xy: Float[Tensor, "line 2"],
    ) -> Float[Tensor, "line 4"]:
        p, _ = xy.shape
        canvas = torch.zeros((p, 4), dtype=xy.dtype, device=xy.device)
        render_points(
            canvas,
            xy,
            starts,
            ends,
            colors,
            widths,
            CAP_TYPES[cap] * torch.ones((p,), dtype=torch.int32, device=xy.device),
            bounds,
            torch.tensor(image_shape, dtype=torch.int64, device=xy.device),
        )
        return canvas

    return color_function
