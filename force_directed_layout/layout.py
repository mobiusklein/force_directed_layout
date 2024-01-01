import math
import warnings

from dataclasses import dataclass, field
from typing import (
    Optional,
    List,
    Any,
    Union,
    Deque,
    Generic,
    TypeVar,
    Callable,
    DefaultDict,
    Dict,
)

import numpy as np

from .quadtree import QuadTree, QuadTreeNode
from .point import VPoint
from .layouts.base import Fn, _ConstFn, jiggle, isnull, ForceLayoutBase
from .layouts.xy import XForceLayout, YForceLayout, RadialForceDirectedLayout
from .layouts.manybody import ManyBodyForcesLayout
from .layouts.linkage import VLinkage, LinkageForceDirectedLayout


@dataclass
class ForceSimulation:
    nodes: List[VPoint] = field(default_factory=list)
    initial_radius: float = 10.0
    initial_angle: float = math.pi * (3 - math.sqrt(5))
    alpha: float = 1.0
    alpha_min: float = 0.001
    alpha_decay: float = 1 - math.pow(alpha_min, 1 / 300)
    alpha_target: float = 0.0
    velocity_decay: float = 0.6
    forces: dict = field(default_factory=dict)
    random: np.random.RandomState = field(
        default_factory=lambda: np.random.RandomState(42)
    )

    def add_force(self, name: str, force: ForceLayoutBase):
        self.forces[name] = force

    def remove_force(self, name: str):
        return self.forces.pop(name)

    def init_nodes(self):
        for i, node in enumerate(self.nodes):
            node.index = i
            if (node.fx is not None) and not node.fixed:
                node.x = node.fx
            if (node.fy is not None) and not node.fixed:
                node.y = node.fy
            if (isnull(node.x) or isnull(node.y)) and not node.fixed:
                radius = self.initial_radius * math.sqrt(0.5 + i)
                angle = self.initial_angle * i
                node.x = radius * math.cos(angle)
                node.y = radius * math.sin(angle)
            if isnull(node.vx) or isnull(node.vy):
                node.vy = node.vx = 0

    def init_forces(self):
        for force in self.forces.values():
            force.initialize()

    def __enter__(self):
        self.init_nodes()
        self.init_forces()
        return self

    def tick(self, iterations: int = 1):
        for k in range(iterations):
            self.alpha += (self.alpha_target - self.alpha) * self.alpha_decay
            for force in self.forces.values():
                force(self.alpha)
            for node in self.nodes:
                if not node.fixed:
                    if node.fx is None:
                        node.vx *= self.velocity_decay
                        node.x += node.vx
                    else:
                        node.x = node.fx
                        node.vx = 0
                    if node.fy is None:
                        node.vy *= self.velocity_decay
                        node.y += node.vy
                    else:
                        node.y = node.fy
                        node.vy = 0
        return self

    def find(self, x, y, radius=None):
        if radius is None:
            radius = float("inf")
        else:
            radius *= radius
        closest = None
        for node in self.nodes:
            dx = x - node.x
            dy = y - node.y
            d2 = dx**2 + dy**2
            if d2 < radius:
                closest = node
                radius = d2
        return closest
