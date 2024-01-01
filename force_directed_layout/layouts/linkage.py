from dataclasses import dataclass
import math
from typing import List, Callable, Dict

import numpy as np

from ..point import VPoint
from .base import ForceLayoutBase, _ConstFn, jiggle, Fn


@dataclass
class VLinkage:
    source: VPoint
    target: VPoint
    index: int = 0

    def distance(self):
        return math.sqrt(
            (self.source.x - self.target.x) ** 2 + (self.source.y - self.target.y) ** 2
        )


class LinkageForceDirectedLayout(ForceLayoutBase):

    nodes: List[VPoint]
    links: List[VLinkage]

    strengths: List[float]
    count: List[float]

    distances: List[float]
    bias: List[float]

    strength: Callable[[VLinkage], float]
    identity: Callable[[VPoint], int]
    distance: Callable[[VPoint], float]
    node_by_id: Dict[int, VPoint]

    def __init__(
        self,
        nodes,
        links,
        strength=None,
        identity=lambda x: x.index,
        distance=Fn(30.0),
    ):
        if strength is None:
            strength = self.default_strength
        self.nodes = nodes
        self.links = links
        self.identity = identity
        self.distance = Fn(distance)
        self.strength = strength

    #         self.initialize()

    def initialize(self, *args, **kwargs):
        if not self.nodes:
            return
        n = len(self.nodes)
        m = len(self.links)
        self.node_by_id = {self.identity(node): node for node in self.nodes}
        self.count = np.zeros(n)

        for i, link in enumerate(self.links):
            link.index = i
            if not isinstance(link.target, (VPoint)):
                link.target = self.node_by_id[link.target]
            if not isinstance(link.source, (VPoint)):
                link.source = self.node_by_id[link.source]
            self.count[link.source.index] += 1
            self.count[link.target.index] += 1

        self.bias = np.zeros(m)
        for i, link in enumerate(self.links):
            self.bias[i] = self.count[link.source.index] / (
                self.count[link.source.index] + self.count[link.target.index]
            )

        self.strengths = np.zeros(m)
        self.distances = np.zeros(m)

        self.init_strengths()
        self.init_distances()

    def init_strengths(self):
        for i, link in enumerate(self.links):
            self.strengths[i] = self.strength(link)

    def init_distances(self):
        for i, link in enumerate(self.links):
            self.distances[i] = self.distance(link)

    def default_strength(self, link):
        return 1 / min(self.count[link.source.index], self.count[link.target.index])

    def force(self, alpha: float):
        for i, link in enumerate(self.links):
            source = link.source
            target = link.target
            x = (target.x + target.vx - source.x - source.vx) or jiggle()
            y = (target.y + target.vy - source.y - source.vy) or jiggle()
            force = math.sqrt(x**2 + y**2)
            force = (force - self.distances[i]) / force * alpha * self.strengths[i]
            x *= force
            y *= force
            b = self.bias[i]
            if not target.fixed:
                target.vx -= x * b
                target.vy -= y * b
            b = 1 - b
            if not source.fixed:
                source.vx += x * b
                source.vy += y * b
