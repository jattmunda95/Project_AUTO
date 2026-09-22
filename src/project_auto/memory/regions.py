"""Pure pixel-space polygon geometry; polygon boundaries count as inside."""

from collections.abc import Sequence
from typing import Protocol

Point = Sequence[float]
Polygon = Sequence[Point]
Box = Sequence[int]


class RegionGeometry(Protocol):
    """Geometry resolver's database-independent view of a configured region."""

    id: int
    polygon: list[list[int]]
    area_px: float


def polygon_area(polygon: Polygon) -> float:
    """Return absolute shoelace area in squared pixels (zero below 3 vertices)."""
    if len(polygon) < 3:
        return 0.0
    return abs(sum(
        polygon[i - 1][0] * point[1] - point[0] * polygon[i - 1][1]
        for i, point in enumerate(polygon)
    )) / 2.0


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray casting for concave or convex polygons; edges and vertices are inside."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for i, (bx, by) in enumerate(polygon):
        ax, ay = polygon[i - 1]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if cross == 0 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def validate_polygon(polygon: Sequence[Sequence[int]]) -> list[list[int]]:
    """Copy a simple, nonzero-area integer polygon, rejecting crossing edges."""
    points = []
    for point in polygon:
        if len(point) != 2 or any(type(value) is not int for value in point):
            raise ValueError("Polygon vertices must be integer [x, y] pairs")
        points.append(list(point))
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len({tuple(point) for point in points}) != len(points):
        raise ValueError("A polygon requires at least three distinct vertices")
    if polygon_area(points) == 0:
        raise ValueError("A polygon must have nonzero area")

    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        # Adjacent collinear edges may continue straight but must not double back.
        previous = points[i - 1]
        if orientation(previous, a, b) == 0 and (
            (previous[0] - a[0]) * (b[0] - a[0])
            + (previous[1] - a[1]) * (b[1] - a[1]) > 0
        ):
            raise ValueError("Polygon edges must not overlap")
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            o1, o2 = orientation(a, b, c), orientation(a, b, d)
            o3, o4 = orientation(c, d, a), orientation(c, d, b)
            if ((o1 * o2 < 0 and o3 * o4 < 0)
                    or (o1 == 0 and on_segment(a, b, c))
                    or (o2 == 0 and on_segment(a, b, d))
                    or (o3 == 0 and on_segment(c, d, a))
                    or (o4 == 0 and on_segment(c, d, b))):
                raise ValueError("Polygon edges must not intersect")
    return points


def resolve_region(box: Box, regions: Sequence[RegionGeometry]) -> int | None:
    """Resolve the bbox centroid; smallest stored area wins, then lowest ID."""
    x1, y1, x2, y2 = box
    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
    matches = [region for region in regions if point_in_polygon(centroid, region.polygon)]
    return min(matches, key=lambda region: (region.area_px, region.id)).id if matches else None


def box_area_fraction(box: Box | None, frame_size: tuple[int, int] | None) -> float | None:
    """Return box area / frame area; frame_size is (width, height)."""
    if frame_size is None:
        return None
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1) / (width * height)
