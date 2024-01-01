import math
from typing import List, Optional, Callable

import numpy as np

from ..point import VPoint
from ..quadtree import QuadTree, QuadTreeNode
from .base import ForceLayoutBase, _ConstFn, jiggle


class ManyBodyForcesLayout(ForceLayoutBase):
    nodes: List[VPoint]
    strengths: List[float]

    current_node: Optional[VPoint] = None
    distance_min2: float = 1
    distance_max2: float = float("inf")
    theta2: float = 0.81
    alpha: float = 1.0

    strength: Callable[[VPoint], float]

    def __init__(self, nodes, strength=_ConstFn(-30)):
        if not callable(strength):
            strength = _ConstFn(strength)
        self.nodes = nodes
        self.strength = strength
        self.initialize()

    def initialize(self, *args, **kwargs):
        self.strengths = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            self.strengths[i] = self.strength(node)

    def force(self, alpha: float, *args, **kwargs):
        tree: QuadTree = QuadTree.from_points(self.nodes)
        tree.visit_after(self.accumulate)
        self.alpha = alpha
        for node in self.nodes:
            if node.fixed:
                continue
            self.current_node = node
            # pre = (node.vx, node.vy)
            tree.visit(self.apply)
            # post = (node.vx, node.vy)
            # print(f"{node.index}, {pre[1]:0.2f} -> {post[1]:0.2f}")

    def accumulate(self, quad: QuadTreeNode, *args, **kwargs):
        strength = 0
        weight = 0
        # For internal nodes, accumulate forces from child quadrants
        if not quad.is_leaf():
            x = y = 0
            for q in quad.children:
                if q is not None:
                    c = abs(q.value)
                    if c:
                        strength += q.value
                        weight += c
                        x += c * q.x
                        y += c * q.y
            quad.x = x / weight
            quad.y = y / weight
        # For leafe nodes, accumulate forces from coincident quadrants
        else:
            for q in quad.points:
                strength += self.strengths[q.index]
            # don't have a concept of "next" q
        quad.value = strength
        return False

    def apply(self, quad: QuadTreeNode, x1, _, x2, *args, **kwargs):
        if not quad.value:
            return True

        x = quad.x - self.current_node.x
        y = quad.y - self.current_node.y
        w = x2 - x1
        force = x**2 + y**2

        # Apply the Barnes-Hut approximation if possible.
        # Limit forces for very close nodes; randomize direction if coincident.
        if w**2 / self.theta2 < force:
            if force < self.distance_max2:
                if x == 0:
                    x = jiggle()
                    force += x**2
                if y == 0:
                    y += jiggle()
                    force += y**2
                if force < self.distance_min2:
                    force = math.sqrt(self.distance_min2 * force)
                self.current_node.vx += x * quad.value * self.alpha / force
                self.current_node.vy += y * quad.value * self.alpha / force
            return True

        # Otherwise process points directly
        elif not quad.points or force > self.distance_max2:
            return

        # TODO: This is incorrect
        # Limit forces for very close nodes, randomize direction if overlap
        # don't have concept of "quad.next"
        if quad is not self.current_node:
            for point in quad.points:
                if x == 0:
                    x = jiggle()
                    force += x**2
                if y == 0:
                    y += jiggle()
                    force += y**2
                if force < self.distance_min2:
                    force = math.sqrt(self.distance_min2 * force)
                w = self.strengths[point.index] * self.alpha / force
                self.current_node.vx += x * w
                self.current_node.vy += y * w
