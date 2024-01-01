import math
from functools import partial
from typing import List, Optional, Callable, Union

import numpy as np

from ..point import VPoint
from ..quadtree import QuadTree, QuadTreeNode
from .base import ForceLayoutBase, _ConstFn, jiggle


class CollisionLayout(ForceLayoutBase):
    nodes: List[VPoint]
    radii: List[float]

    radius: Callable[[VPoint, int, List[VPoint]], float]
    strength: Callable[[VPoint], float]

    def __init__(
        self, nodes: List[VPoint], radius: Callable, strength: Callable = _ConstFn(1.0)
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.radius = radius
        self.strength = strength

    def initialize(self, *args, **kwargs):
        self.radii = np.zeros(len(self.nodes))
        for node in self.nodes:
            self.radii[node.index] = self.radius(node, node.index, self.nodes)

    def force(self, *args, **kwargs):
        tree = QuadTree.from_points(self.nodes)
        tree.visit_after(self.prepare)
        for node in self.nodes:
            ri = self.radii[node.index]
            xi = node.x + node.vx
            yi = node.y + node.vy
            tree.visit(partial(self.apply, node, xi, yi, ri))

    def radius_of(self, x):
        if isinstance(x, VPoint):
            return self.radii[x.index]
        elif isinstance(x, QuadTreeNode):
            return x.data.get("radius")
        else:
            return 0.0

    def prepare(self, quad: QuadTreeNode, *args, **kwargs):
        if quad.is_leaf():
            if quad.points:
                quad.data["radius"] = max(map(self.radius_of, quad.points))
            else:
                quad.data["radius"] = 0.0
        else:
            quad.data["radius"] = max(map(self.radius_of, quad.children))

    def apply(
        self,
        node: VPoint,
        xi: float,
        yi: float,
        ri: float,
        quad: Union[VPoint, QuadTreeNode],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ):
        rj = self.radius_of(quad)
        r = rj + ri
        if isinstance(quad, QuadTreeNode) and quad.is_leaf():
            for pt in quad.points:
                self.apply(node, xi, yi, ri, pt, x0, y0, x1, y1)
        if isinstance(quad, VPoint):
            if quad.index > node.index:
                x = xi - quad.x - quad.vx
                y = yi - quad.y - quad.vy
                li = x**2 + y**2
                rad_hit = li < r**2
                ov_hit = quad.overlaps(node)
                if rad_hit or ov_hit:
                    # if not rad_hit and ov_hit:
                    #     print(f"Overlap Hit {node} & {quad}")
                    if x == 0:
                        x = jiggle()
                        li += x**2
                    if y == 0:
                        y = jiggle()
                        li += y**2
                    li = math.sqrt(li)
                    li = (r - li) / li * self.strength(node)
                    x *= li
                    y *= li
                    rj *= rj
                    r = rj / (ri**2 + rj)
                    if not node.fixed:
                        node.vx += x * r
                        node.vy += y * r
                    r = 1 - r
                    if not quad.fixed:
                        quad.vx -= x * r
                        quad.vy -= y * r
                return
        return x0 > xi + r or x1 < xi - r or y0 > yi + r or y1 < yi - r
