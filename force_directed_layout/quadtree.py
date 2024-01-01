from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Union, Deque, Generic, TypeVar, Callable


@dataclass
class Point:
    x: float
    y: float
    data: Optional[dict] = field(default_factory=dict)

    def as_quadrant(self, width: float, height: float):
        return Quadrant(self.data, self.x, self.y, width, height)


class _QuadrantMixin:
    __slots__ = ()

    def contains(self, point: Point):
        return (
            self.x <= point.x < self.x + self.width
            and self.y <= point.y < self.y + self.height
        )

    def intersects(self, x, y, width, height):
        return (
            self.x < x + width
            and x < self.x + self.width
            and self.y < y + height
            and y < self.y + self.height
        )

    def intersects_quad(self, other: "Quadrant"):
        return self.intersects(other.x, other.y, other.width, other.height)


@dataclass
class Quadrant(_QuadrantMixin):
    data: Any
    x: float
    y: float
    width: float
    height: float

    def intersects_quad(self, other: "Quadrant"):
        return self.intersects(other.x, other.y, other.width, other.height)

    def pad(self, x: float = 0.0, y: float = 0.0):
        return self.__class__(
            self.data, self.x - x, self.y - y, self.width + x, self.height + y
        )


class QuadTreeNode(_QuadrantMixin):
    x: float
    y: float
    width: float
    height: float

    level: int = 0
    points: List[Point]
    children: List[Optional["QuadTreeNode"]]
    data: Dict[str, Any]

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.points = []
        self.children = [None, None, None, None]  # NW, NE, SW, SE
        self.data = {}

    def __repr__(self):
        t = f"{self.__class__.__name__}({self.x}, {self.y}, {self.width}, {self.height}, <{len(self.points)} points>)"
        return t

    def is_leaf(self):
        return not any(self.children)

    def insert(self, point: Point) -> bool:
        if not self.contains(point):
            return False

        if (
            self.is_leaf()
            and len(self.points) < 4
            or self.width <= 1
            or self.height <= 1
        ):
            self.points.append(point)
            return True
        else:
            if self.children[0] is None:
                self.subdivide()
            status = False
            for i, child in enumerate(self.children):
                status |= child.insert(point)
            return status

    def subdivide(self):
        half_width = self.width / 2
        half_height = self.height / 2
        x, y = self.x, self.y
        self.children[0] = QuadTreeNode(x, y, half_width, half_height)  # NW
        self.children[1] = QuadTreeNode(
            x + half_width, y, half_width, half_height
        )  # NE
        self.children[2] = QuadTreeNode(
            x, y + half_height, half_width, half_height
        )  # SW
        self.children[3] = QuadTreeNode(
            x + half_width, y + half_height, half_width, half_height
        )  # SE
        for child in self.children:
            child.level += 1
        # Reinsert points into children
        for point in self.points:
            for child in self.children:
                if child.contains(point):
                    child.insert(point)
        self.points = []

    def contains(self, point):
        return self.x <= point.x < (self.x + self.width) and self.y <= point.y < (
            self.y + self.height
        )

    def query_range_quad(self, quad: Quadrant):
        results = []
        if not self.intersects_quad(quad):
            return results

        for point in self.points:
            if quad.contains(point):
                results.append(point)
        if not self.is_leaf():
            for child in self.children:
                if child:
                    results.extend(child.query_range_quad(quad))
        return results

    def query_range(self, x, y, width, height):
        quad = Quadrant(None, x, y, width, height)
        results = self.query_range_quad(quad)
        return results


class QuadTree:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.root = QuadTreeNode(x, y, width, height)

    def expand_to_cover(self, point: Point):
        if self.root.contains(point):
            return False
        _old_points = self.root.points

        old_end_x = self.root.x + self.root.width
        old_end_y = self.root.y + self.root.height

        if point.x < self.root.x:
            self.root.x = point.x - abs(self.root.width) // 2
            self.root.width = old_end_x - self.root.x + 1
        elif point.x > old_end_x:
            self.root.width = point.x + abs(self.root.width) // 2 + 1

        if point.y < self.root.y:
            self.root.y = point.y - abs(self.root.height) // 2
            self.root.height = old_end_y - self.root.y + 1
        elif point.y > old_end_y:
            self.root.height = point.y + abs(self.root.height) // 2 + 1

        self.root.children = [None, None, None, None]
        self.root.subdivide()
        return True

    def insert(self, point: Point, expand: bool = True) -> bool:
        inserted = self.root.insert(point)
        if not inserted and expand:
            self.expand_to_cover(point)
            inserted = self.root.insert(point)
        return inserted

    def insert_all(self, points: List[Point], expand: bool = True) -> bool:
        for point in points:
            self.insert(point, expand)

    def query_range(self, x, y, width, height):
        return self.root.query_range(x, y, width, height)

    def query_range_quad(self, quad: Quadrant):
        return self.root.query_range_quad(quad)

    def query_point(self, x, y):
        pt = Point(x, y)
        node = self.root
        if not node.contains(pt):
            return None
        while not node.is_leaf():
            for child in node.children:
                if child and child.contains(pt):
                    node = child
                    break
            else:
                break
        return node

    def visit(self, callback):
        quads = Deque()
        node = self.root

        if node:
            quads.append(
                Quadrant(
                    node,
                    node.x,
                    node.y,
                    node.width,
                    node.height,
                )
            )
        while quads:
            q = quads.pop()
            node = q.data
            x = q.x
            y = q.y
            width = q.width
            height = q.height

            if not callback(node, x, y, width, height) and node.children:
                for child_maybe in reversed(node.children):
                    if child_maybe:
                        quads.append(
                            Quadrant(
                                child_maybe,
                                child_maybe.x,
                                child_maybe.y,
                                child_maybe.width,
                                child_maybe.height,
                            )
                        )
        return self

    def visit_after(self, callback):
        """Visit quadrants in reverse order"""
        quads = Deque()
        node = self.root
        acc = []
        if node:
            quads.append(
                Quadrant(
                    node,
                    node.x,
                    node.y,
                    node.width,
                    node.height,
                )
            )
        while quads:
            q = quads.pop()
            node = q.data
            if node.children:
                for child_maybe in node.children:
                    if child_maybe:
                        quads.append(
                            Quadrant(
                                child_maybe,
                                child_maybe.x,
                                child_maybe.y,
                                child_maybe.width,
                                child_maybe.height,
                            )
                        )
            acc.append(q)
        for q in reversed(acc):
            callback(q.data, q.x, q.y, q.width, q.height)
        return self

    @property
    def x(self):
        return self.root.x

    @property
    def y(self):
        return self.root.y

    @property
    def width(self):
        return self.root.width

    @property
    def height(self):
        return self.root.height

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.width}, {self.height})"

    @classmethod
    def extents(cls, points: List[Point]):
        inf = float("inf")
        min_x = inf
        max_x = -inf
        min_y = inf
        max_y = -inf

        for p in points:
            if p.x < min_x:
                min_x = p.x
            if p.x > max_x:
                max_x = p.x
            if p.y < min_y:
                min_y = p.y
            if p.y > max_y:
                max_y = p.y
        width = max(max_x - min_x, 2.0) + 1
        height = max(max_y - min_y, 2.0) + 1
        return min_x, min_y, width, height

    @classmethod
    def from_points(cls, points: List[Point]):
        (min_x, min_y, width, height) = cls.extents(points)
        self = cls(min_x - 1, min_y - 1, width + 1, height + 1)
        for p in points:
            assert self.insert(p)
        return self
