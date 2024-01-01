from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from force_directed_layout.quadtree import Quadrant, Point


class BBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    anchor: Point

    def expand(self, xmin, ymin, xmax, ymax) -> "BBox":
        return self.__class__(
            min(xmin, self.xmin),
            min(ymin, self.ymin),
            max(xmax, self.xmax),
            max(ymax, self.ymax),
            self.anchor,
        )

    def center(self):
        xc = self.anchor.x
        yc = self.anchor.y
        return BBox(
            self.xmin - xc, self.ymin - yc, self.xmax - xc, self.ymax - yc, Point(0, 0)
        )

    def scale(self, xscale: float, yscale: float):
        return BBox(
            self.xmin * xscale,
            self.ymin * yscale,
            self.xmax * xscale,
            self.ymax * yscale,
            Point(self.anchor.x * xscale, self.anchor.y * yscale),
        )

    def relative_to_point(self, x, y=None):
        ref = self.center()
        if y is None and hasattr(x, "x"):
            y = x.y
            x = x.x
        bb = BBox(
            ref.xmin + x,
            ref.ymin + y,
            ref.xmax + x,
            ref.ymax + y,
            Point(ref.anchor.x + x, ref.anchor.y + y),
        )
        return bb
        # ex = bb.anchor.x + x
        # ey = bb.anchor.y + y
        # return bb.expand(ex, ey, ex, ey)

    def as_array(self):
        return np.array(self).reshape((-1, 2))

    def as_quadrant(self, data=None):
        return Quadrant(
            data, self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin
        )

    def contains(self, xy: Point):
        return self.as_quadrant().contains(xy)

    def intersects(self, other: "BBox"):
        return self.as_quadrant().intersects_quad(other.as_quadrant())


@dataclass
class VPoint:
    x: float
    y: float
    vx: float = None
    vy: float = None
    fx: float = None
    fy: float = None
    index: int = 0
    fixed: bool = False
    data: Optional[dict] = field(default_factory=dict)
    bounds: Optional[BBox] = None

    def as_quadrant(self, width: float = None, height: float = None):
        if width is None or height is None and self.bounds:
            return self.bounds.relative_to_point(self.x, self.y).as_quadrant()
        return Quadrant(self.data, self.x, self.y, width, height)

    def overlaps(self, other: "VPoint") -> bool:
        if self.bounds:
            quad = self.as_quadrant()
            if other.bounds:
                other = other.as_quadrant()
                return quad.intersects_quad(other)
            else:
                return quad.contains(other)
        elif other.bounds:
            return other.as_quadrant().contains(self)
        else:
            return False


#     @property
#     def x(self):
#         return self._x

#     @property
#     def y(self):
#         return self._y

#     @x.setter
#     def x(self, value):
#         warnings.warn(f"Setting x@{self.index} to {value}", stacklevel=2)
#         self._x = value

#     @x.setter
#     def y(self, value):
#         warnings.warn(f"Setting y@{self.index} to {value}", stacklevel=2)
#         self._y = value
