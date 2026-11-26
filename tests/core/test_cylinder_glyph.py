"""Tests for batched cylinder glyphs with independent length and radius."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv


@pytest.fixture
def two_orthogonal_glyphs():
    """Two fixed samples pointing along x and y with distinct sizes."""
    points = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    mesh = pv.PolyData(points)
    mesh['direction'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh['length'] = np.array([2.0, 3.0])
    mesh['radius'] = np.array([0.5, 0.25])
    mesh['color'] = np.array([10.0, 20.0])
    return mesh


def test_cylinder_glyph_dimensions_directions_and_batch_size(two_orthogonal_glyphs):
    mesh = two_orthogonal_glyphs
    reference = pv.Cylinder(
        direction=(1.0, 0.0, 0.0),
        radius=1.0,
        height=1.0,
        resolution=20,
    )

    glyphs = mesh.cylinder_glyph(
        direction='direction',
        length='length',
        radius='radius',
        resolution=20,
    )

    assert isinstance(glyphs, pv.PolyData)
    assert glyphs.n_points == 2 * reference.n_points
    assert glyphs.n_cells == 2 * reference.n_cells

    # x cylinder spans x in [-1, 1]; the y cylinder is centered at y=4 with
    # half-length 1.5 and radius 0.25, so y spans [-0.5, 5.5].
    expected_bounds = (-1.0, 1.0, -0.5, 5.5, -0.5, 0.5)
    assert glyphs.bounds == pytest.approx(expected_bounds, abs=1e-6)


def test_cylinder_glyph_accepts_arrays_and_preserves_scalars(two_orthogonal_glyphs):
    mesh = two_orthogonal_glyphs

    glyphs = mesh.cylinder_glyph(
        mesh['direction'],
        mesh['length'],
        mesh['radius'],
        scalars='color',
        resolution=20,
    )

    assert glyphs.active_scalars_name == 'color'
    assert np.allclose(np.unique(glyphs['color']), [10.0, 20.0])

    # Every copied source point of each glyph carries that glyph's scalar, in
    # input-point order.
    points_per_glyph = glyphs.n_points // mesh.n_points
    assert np.all(glyphs['color'][:points_per_glyph] == 10.0)
    assert np.all(glyphs['color'][points_per_glyph:] == 20.0)


def test_cylinder_glyph_uses_active_scalars_by_default(two_orthogonal_glyphs):
    mesh = two_orthogonal_glyphs
    mesh.set_active_scalars('color')

    glyphs = mesh.cylinder_glyph(
        direction='direction',
        length='length',
        radius='radius',
        resolution=12,
    )

    assert glyphs.active_scalars_name == 'color'
    assert np.allclose(np.unique(glyphs['color']), [10.0, 20.0])


def test_cylinder_glyph_accepts_direct_scalar_array():
    mesh = pv.PolyData(np.zeros((2, 3)))
    glyphs = mesh.cylinder_glyph(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [2.0, 2.0],
        [0.25, 0.25],
        scalars=np.array([1, 2]),
        resolution=12,
    )
    assert glyphs.active_scalars_name == '_cylinder_glyph_scalars'
    assert set(np.unique(glyphs['_cylinder_glyph_scalars'])) == {1, 2}


def test_cylinder_glyph_broadcasts_scalar_values():
    mesh = pv.PolyData(np.zeros((4, 3)))
    glyphs = mesh.cylinder_glyph([1.0, 0.0, 0.0], 2.0, 0.5, resolution=8)

    reference = pv.Cylinder(direction=(1.0, 0.0, 0.0), resolution=8)
    assert glyphs.n_points == 4 * reference.n_points
    assert glyphs.n_cells == 4 * reference.n_cells
    # All four coincident glyphs still share the single-glyph bounds.
    assert glyphs.bounds == pytest.approx((-1.0, 1.0, -0.5, 0.5, -0.5, 0.5))


def test_cylinder_glyph_batch_scales_linearly():
    n_glyphs = 25
    mesh = pv.PolyData(
        np.column_stack([np.arange(n_glyphs), np.zeros(n_glyphs), np.zeros(n_glyphs)])
    )
    glyphs = mesh.cylinder_glyph(
        [1.0, 0.0, 0.0],
        1.0,
        0.1,
        resolution=8,
    )
    reference = pv.Cylinder(direction=(1.0, 0.0, 0.0), resolution=8)
    assert glyphs.n_points == n_glyphs * reference.n_points
    assert glyphs.n_cells == n_glyphs * reference.n_cells


def test_cylinder_glyph_start_anchor():
    mesh = pv.PolyData(np.zeros((1, 3)))
    glyphs = mesh.cylinder_glyph(
        [1.0, 0.0, 0.0],
        2.0,
        0.5,
        resolution=20,
        centered=False,
    )
    assert glyphs.bounds == pytest.approx((0.0, 2.0, -0.5, 0.5, -0.5, 0.5), abs=1e-6)


def test_cylinder_glyph_factor_scales_both_dimensions():
    mesh = pv.PolyData(np.zeros((1, 3)))
    glyphs = mesh.cylinder_glyph(
        [1.0, 0.0, 0.0],
        2.0,
        0.5,
        factor=2.0,
        resolution=20,
    )
    assert glyphs.bounds == pytest.approx((-2.0, 2.0, -1.0, 1.0, -1.0, 1.0), abs=1e-6)


def test_cylinder_glyph_empty_input():
    glyphs = pv.PolyData().cylinder_glyph()
    assert isinstance(glyphs, pv.PolyData)
    assert glyphs.n_points == 0
    assert glyphs.n_cells == 0


def test_cylinder_glyph_zero_length_is_degenerate():
    mesh = pv.PolyData(np.zeros((1, 3)))
    glyphs = mesh.cylinder_glyph([1.0, 0.0, 0.0], 0.0, 0.5, resolution=20)
    assert glyphs.bounds == pytest.approx((0.0, 0.0, -0.5, 0.5, -0.5, 0.5), abs=1e-9)
    assert glyphs.n_points > 0


def test_cylinder_glyph_zero_radius_collapses_to_axis():
    mesh = pv.PolyData(np.zeros((1, 3)))
    glyphs = mesh.cylinder_glyph([1.0, 0.0, 0.0], 2.0, 0.0, resolution=20)
    assert glyphs.bounds == pytest.approx((-1.0, 1.0, 0.0, 0.0, 0.0, 0.0), abs=1e-9)


def test_cylinder_glyph_zero_direction_is_invalid():
    mesh = pv.PolyData(np.zeros((1, 3)))
    with pytest.raises(ValueError, match='nonzero magnitude'):
        mesh.cylinder_glyph([[0.0, 0.0, 0.0]], [1.0], [0.1])


@pytest.mark.parametrize(
    ('length', 'radius', 'match'),
    [
        (np.array([-1.0, 1.0]), np.array([0.1, 0.1]), 'length'),
        (np.array([1.0, 1.0]), np.array([-0.1, 0.1]), 'radius'),
        (np.array([np.nan, 1.0]), np.array([0.1, 0.1]), 'finite'),
        (np.array([1.0, 1.0]), np.array([0.1, np.nan]), 'finite'),
        (np.array([np.inf, 1.0]), np.array([0.1, 0.1]), 'finite'),
    ],
)
def test_cylinder_glyph_invalid_values(length, radius, match):
    mesh = pv.PolyData(np.zeros((2, 3)))
    with pytest.raises(ValueError, match=match):
        mesh.cylinder_glyph([[1.0, 0.0, 0.0]] * 2, length, radius)


def test_cylinder_glyph_wrong_direction_shape():
    mesh = pv.PolyData(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="'direction' must have shape"):
        mesh.cylinder_glyph([[1.0, 0.0, 0.0]], [1.0, 1.0], [0.1, 0.1])


@pytest.mark.parametrize('array_name', ['missing_direction', 'missing_length', 'missing_radius'])
def test_cylinder_glyph_missing_array(array_name):
    mesh = pv.PolyData(np.zeros((1, 3)))
    mesh['length'] = [1.0]
    mesh['radius'] = [0.1]
    kwargs = {
        'direction': [1.0, 0.0, 0.0],
        'length': 'length',
        'radius': 'radius',
    }
    if array_name == 'missing_direction':
        kwargs['direction'] = array_name
    elif array_name == 'missing_length':
        kwargs['length'] = array_name
    else:
        kwargs['radius'] = array_name
    with pytest.raises(KeyError, match=array_name):
        mesh.cylinder_glyph(**kwargs)


def test_glyph_length_and_radius_keywords_dispatch():
    mesh = pv.PolyData(np.zeros((1, 3)))
    glyphs = mesh.glyph(
        orient=np.array([[1.0, 0.0, 0.0]]),
        length=[2.0],
        radius=[0.5],
    )
    assert isinstance(glyphs, pv.PolyData)
    assert glyphs.bounds == pytest.approx((-1.0, 1.0, -0.5, 0.5, -0.5, 0.5), abs=1e-6)


def test_glyph_without_length_or_radius_keeps_uniform_semantics():
    # The default glyph is still the arrow-based uniform vtkGlyph3D path.
    mesh = pv.PolyData(np.zeros((1, 3)))
    mesh['vectors'] = np.array([[1.0, 0.0, 0.0]])
    arrows = mesh.glyph(orient='vectors', scale=False)
    assert isinstance(arrows, pv.PolyData)
    assert arrows.n_points > 0
    # An arrow is not a unit-radius cylinder: it extends along x well beyond
    # the cylinder's half-length for the same nominal factor.
    assert arrows.bounds.x_max > 0.5
