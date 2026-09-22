"""Geometry contracts independent of SQLAlchemy and camera hardware."""

from dataclasses import dataclass

import pytest

from project_auto.memory.regions import (
    box_area_fraction, point_in_polygon, polygon_area, resolve_region, validate_polygon,
)

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


@dataclass
class Geometry:
    id: int
    polygon: list[list[int]]
    area_px: float


@pytest.mark.parametrize("polygon, expected", [
    (SQUARE, 10000),
    (list(reversed(SQUARE)), 10000),
    ([[0, 0], [6, 0], [6, 2], [2, 2], [2, 5], [0, 5]], 18),
    ([], 0),
])
def test_polygon_area(polygon, expected):
    assert polygon_area(polygon) == expected
    assert isinstance(polygon_area(polygon), float)


@pytest.mark.parametrize("point, expected", [
    ((50, 50), True), ((101, 50), False), ((99.999, 50), True),
    ((100.001, 50), False), ((100, 50), True), ((0, 0), True),
])
def test_polygon_contains_boundary(point, expected):
    assert point_in_polygon(point, SQUARE) is expected
    assert point_in_polygon(point, list(reversed(SQUARE))) is expected


def test_concave_polygon():
    polygon = [[0, 0], [6, 0], [6, 2], [2, 2], [2, 5], [0, 5]]
    assert point_in_polygon((1, 4), polygon)
    assert not point_in_polygon((4, 4), polygon)


def test_resolve_uses_centroid_and_smallest_area():
    large = Geometry(1, SQUARE, 10000)
    small = Geometry(2, [[20, 20], [70, 20], [70, 60], [20, 60]], 2000)
    assert resolve_region((0, 0, 90, 80), [large, small]) == 2
    assert resolve_region((0, 0, 90, 80), [small, large]) == 2
    assert resolve_region((80, 80, 90, 90), [large, small]) == 1
    assert resolve_region((90, 90, 200, 200), [large, small]) is None
    assert resolve_region((0, 0, 10, 10), []) is None
    assert resolve_region((0, 0, 90, 80), [Geometry(3, SQUARE, 10000), large]) == 1


@pytest.mark.parametrize("polygon", [
    [], [[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1.5, 0], [1, 2]], [[0, 0], [True, 0], [1, 2]],
    [[0, 0], [6, 4], [0, 4], [4, 0]],
    [[0, 0], [4, 0], [2, 0], [2, 4], [0, 4]],
])
def test_invalid_polygon(polygon):
    with pytest.raises(ValueError):
        validate_polygon(polygon)


def test_closed_polygon_is_normalized_and_copied():
    result = validate_polygon(SQUARE + [SQUARE[0]])
    assert result == SQUARE
    assert result[0] is not SQUARE[0]


def test_normalized_area():
    assert box_area_fraction((0, 0, 10, 20), (100, 100)) == 0.02
    assert box_area_fraction((10, 0, 0, 20), (100, 100)) == 0
    assert box_area_fraction(None, (100, 100)) is None
    assert box_area_fraction((0, 0, 10, 20), None) is None
    with pytest.raises(ValueError):
        box_area_fraction((0, 0, 10, 20), (0, 100))
