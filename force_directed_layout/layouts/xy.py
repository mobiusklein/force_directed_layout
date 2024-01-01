import math
from typing import List, Callable

import numpy as np

from ..point import VPoint
from .base import ForceLayoutBase, isnull, Fn


class XForceLayout(ForceLayoutBase):
    nodes: List[VPoint]
    x: Callable[[VPoint], float] = Fn(0.0)
    strength: Callable[[VPoint], float] = Fn(0.1)
    strengths: List[float]
    xz: List[float]

    def __init__(self, nodes: List[VPoint], x=Fn(0.0), strength=Fn(0.1)):
        x = Fn(x)
        self.nodes = nodes
        self.strength = strength
        self.x = x
        self.initialize()

    def initialize(self, *args, **kwargs):
        self.xz = np.zeros(len(self.nodes))
        self.strengths = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            self.xz[i] = self.x(node, i, self.nodes)
            self.strengths[i] = (
                0 if isnull(self.xz[i]) else self.strength(node, i, self.nodes)
            )

    def force(self, alpha: float):
        for i, node in enumerate(self.nodes):
            if node.fixed:
                continue
            node.vx += (self.xz[i] - node.x) * self.strengths[i] * alpha

    def __call__(self, alpha: float):
        self.force(alpha)


class YForceLayout(ForceLayoutBase):
    nodes: List[VPoint]
    y: Callable[[VPoint, int, List[VPoint]], float] = Fn(0.0)
    strength: Callable[[VPoint], float] = Fn(0.1)
    strengths: List[float]
    yz: List[float]

    def __init__(self, nodes: List[VPoint], y=Fn(0.0), strength=Fn(0.1)):
        y = Fn(y)
        self.nodes = nodes
        self.strength = strength
        self.y = y
        self.initialize()

    def initialize(self, *args, **kwargs):
        self.yz = np.zeros(len(self.nodes))
        self.strengths = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            self.yz[i] = self.y(node, i, self.nodes)
            self.strengths[i] = (
                0 if isnull(self.yz[i]) else self.strength(node, i, self.nodes)
            )

    def force(self, alpha: float):
        for i, node in enumerate(self.nodes):
            if node.fixed:
                continue
            node.vy += (self.yz[i] - node.y) * self.strengths[i] * alpha

    def __call__(self, alpha: float):
        self.force(alpha)


class RadialForceDirectedLayout(ForceLayoutBase):
    nodes: List[VPoint]
    strengths: List[float]
    radii: List[float]

    radius: Callable[[VPoint, int, List[VPoint]], float] = Fn(1.0)
    strength: Callable[[VPoint, int, List[VPoint]], float] = Fn(0.1)

    x: float = 0.0
    y: float = 0.0

    def __init__(
        self,
        nodes: List[VPoint],
        x=0.0,
        y=0.0,
        strength=Fn(0.1),
        radius=Fn(1.0),
    ):
        self.nodes = nodes
        self.strength = strength
        self.radius = radius
        self.x = x
        self.y = y
        self.initialize()

    def initialize(self, *args, **kwargs):
        self.radii = [
            self.radius(node, i, self.nodes) for i, node in enumerate(self.nodes)
        ]
        self.strengths = [
            self.strength(node, i, self.nodes) for i, node in enumerate(self.nodes)
        ]

    def force(self, alpha: float):
        for i, node in enumerate(self.nodes):
            if node.fixed:
                continue
            dx = node.x - (self.x or 1e-6)
            dy = node.y - (self.y or 1e-6)
            r = math.sqrt(dx**2 + dy**2)
            k = (self.radii[i] - r) * self.strengths[i] * alpha / r
            node.vx += dx * k
            node.vy += dy * k

    def __call__(self, alpha: float):
        self.force(alpha)
