from __future__ import annotations

import re
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.errors import PyVistaPickingError

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


def test_single_cell_picking():
    sphere = pv.Sphere()
    width, height = 100, 100

    class PickCallback:
        def __init__(self):
            self.called = False

        def __call__(self, *args, **kwargs):  # noqa: ARG002
            self.called = True

    pl = pv.Plotter(
        window_size=(width, height),
    )

    callback = PickCallback()
    pl.enable_cell_picking(
        start=False,
        show=True,
        callback=callback,
        through=False,  # Single cell visible picking
    )
    pl.add_mesh(sphere)
    pl.show(auto_close=False)  # must start renderer first

    width, height = pl.window_size
    pl.iren._mouse_move(width // 2, height // 2)
    pl.iren._simulate_keypress('p')

    pl.close()

    assert callback.called
    assert isinstance(pl.picked_cells, pv.UnstructuredGrid)
    assert pl.picked_cells.n_cells == 1


def test_picked_cells_attribute():
    """Test the ``picked_cells`` attribute when selecting through multiple cells."""

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere(), pickable=True)
    pl.add_mesh(pv.Sphere().translate((1, 0, 0)), pickable=True)
    pl.add_mesh(pv.Sphere().translate((2, 0, 0)), pickable=True)
    pl.enable_cell_picking(through=True)
    pl.view_xy()
    pl.show(auto_close=False)  # must start renderer first

    width, height = pl.window_size

    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width // 2, height)

    assert isinstance(pl.picked_cells, pv.MultiBlock)
    assert len(pl.picked_cells) == 2

    pl.iren._mouse_left_button_press(width // 2, 0)
    pl.iren._mouse_left_button_release(width, height)

    assert isinstance(pl.picked_cells, pv.MultiBlock)
    assert len(pl.picked_cells) == 2

    pl.iren._mouse_left_button_press(width // 3, 0)
    pl.iren._mouse_left_button_release(2 * width // 3, height)

    assert isinstance(pl.picked_cells, pv.MultiBlock)
    assert len(pl.picked_cells) == 3

    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width // 6, height)

    assert isinstance(pl.picked_cells, pv.UnstructuredGrid)


@pytest.mark.parametrize('through', [False, True])
def test_multi_cell_picking(through):
    cube = pv.Cube()

    # Test with algorithm source to make sure connections work with picking
    src = _vtk.vtkSphereSource()
    src.SetCenter((1, 0, 0))
    mapper = _vtk.vtkDataSetMapper()
    mapper.SetInputConnection(src.GetOutputPort())
    actor = _vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(True)

    pl = pv.Plotter(window_size=(1024, 768))
    pl.add_mesh(cube, pickable=True)
    pl.add_actor(actor)
    pl.enable_cell_picking(
        color='blue',
        through=through,
        start=True,
        show=True,
        show_frustum=True,
    )
    pl.show(auto_close=False)  # must start renderer first

    # simulate the pick (169, 113, 875, 684)
    pl.iren._mouse_left_button_press(169, 113)
    pl.iren._mouse_move(875, 684)
    pl.iren._mouse_left_button_release()

    assert pl.get_pick_position() == (169, 113, 875, 684)

    pl.close()

    assert isinstance(pl.picked_cells, pv.MultiBlock)
    # Selection should return 2 submeshes
    assert len(pl.picked_cells) == 2

    assert all('original_cell_ids' in b.cell_data for b in pl.picked_cells)
    assert all('orig_extract_id' not in b.cell_data for b in pl.picked_cells)

    merged = pl.picked_cells.combine()
    n_sphere_cells = pv.wrap(src.GetOutput()).n_cells
    if through:
        # all cells should have been selected
        assert merged.n_cells == cube.n_cells + n_sphere_cells
    else:
        assert merged.n_cells < cube.n_cells + n_sphere_cells


@pytest.mark.parametrize('left_clicking', [False, True])
def test_mesh_picking(sphere, left_clicking):
    picked = []

    def callback(picked_mesh):
        picked.append(picked_mesh)

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere)
    pl.enable_mesh_picking(callback=callback, left_clicking=left_clicking)
    pl.show(auto_close=False)

    width, height = pl.window_size

    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert sphere in picked
    assert pl.picked_mesh == sphere
    assert pl.picked_actor == actor

    # invalid selection
    if left_clicking:
        pl.iren._mouse_left_button_click(0, 0)
    else:
        pl.iren._mouse_right_button_click(0, 0)

    assert pl.picked_mesh is None


def test_actor_picking(sphere):
    picked = []

    def callback(picked_actor):
        picked.append(picked_actor)

    pl = pv.Plotter()
    actor = pl.add_mesh(sphere)
    pl.enable_mesh_picking(callback=callback, use_actor=True)
    pl.show(auto_close=False)

    width, height = pl.window_size

    pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert actor in picked
    assert pl.picked_mesh == sphere

    # invalid selection
    pl.iren._mouse_right_button_click(0, 0)

    assert pl.picked_mesh is None


@pytest.mark.parametrize('left_clicking', [False, True])
def test_surface_point_picking(sphere, left_clicking):
    picked = []

    def callback(point):
        picked.append(point)

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.enable_surface_point_picking(callback=callback, left_clicking=left_clicking)
    pl.show(auto_close=False)

    width, height = pl.window_size

    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert picked
    assert pl.picked_point is not None

    # invalid selection
    if left_clicking:
        pl.iren._mouse_left_button_click(0, 0)
    else:
        pl.iren._mouse_right_button_click(0, 0)

    assert pl.picked_point is None


@pytest.mark.parametrize('left_clicking', [False, True])
def test_disable_picking(sphere, left_clicking):
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.enable_surface_point_picking(left_clicking=left_clicking)
    pl.disable_picking()
    pl.show(auto_close=False)

    assert pl.picking._picking_text not in pl.renderer.actors

    width, height = pl.window_size

    # This verifies callbacks are removed from click events
    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert pl.picked_point is None

    pl.disable_picking()  # ensure it can safely be called twice


def test_cell_picking_interactive():
    n_cells = []

    def callback(picked_cells):
        n_cells.append(picked_cells.n_cells)

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.enable_cell_picking(callback=callback)
    pl.show(auto_close=False, interactive=False)

    width, height = pl.window_size

    # simulate "r" keypress
    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(width // 2, height // 2)
    pl.iren._mouse_left_button_release(width, height)

    assert n_cells[0]
    assert pl.picked_cells
    assert 'original_cell_ids' in pl.picked_cells.cell_data
    assert 'orig_extract_id' not in pl.picked_cells.cell_data


def test_cell_picking_original_ids():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.enable_cell_picking(through=True)
    pl.show(auto_close=False, interactive=False)

    width, height = pl.window_size

    # simulate "r" keypress
    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(width // 2, height // 2)
    pl.iren._mouse_left_button_release(width, height)

    assert pl.picked_cells
    assert 'original_cell_ids' in pl.picked_cells.cell_data
    assert 'orig_extract_id' not in pl.picked_cells.cell_data


def test_picked_cell_removed():
    pl = pv.Plotter()
    assert not hasattr(pl, 'picked_cell')


def _display_position(pl, point):
    """Project a world point to display coordinates in the active renderer."""
    coordinate = _vtk.vtkCoordinate()
    coordinate.SetValue(*point)
    return coordinate.GetComputedDisplayValue(pl.renderer)


def test_rectangle_point_picking_ids_and_coordinates():
    """A rectangle selects the source IDs and world coordinates of visible points."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = pv.PolyData(points)

    picked_results = []
    pl = pv.Plotter(window_size=(400, 400))
    pl.add_mesh(mesh, point_size=15)
    pl.enable_rectangle_point_picking(
        callback=picked_results.append, show=False, show_message=False
    )
    pl.view_xy()
    pl.show(auto_close=False, interactive=False)
    width, height = pl.window_size

    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width - 1, height - 1)

    assert len(picked_results) == 1
    picked = picked_results[0]
    assert isinstance(picked, pv.PolyData)
    assert pl.picked_points is picked
    assert picked.n_points == 4
    np.testing.assert_array_equal(picked.point_data['original_point_ids'], np.arange(4))
    np.testing.assert_allclose(picked.points, points)

    # A rectangle around a single point returns only that point's ID
    display = _display_position(pl, points[2])
    pl.iren._mouse_left_button_press(display[0] - 8, display[1] - 8)
    pl.iren._mouse_left_button_release(display[0] + 8, display[1] + 8)
    single = pl.picked_points
    assert isinstance(single, pv.PolyData)
    assert single.n_points == 1
    assert single.point_data['original_point_ids'].tolist() == [2]
    np.testing.assert_allclose(single.points, points[2:3])
    pl.close()


def test_rectangle_point_picking_occlusion():
    """Points hidden behind a visible surface are not picked."""
    plane = pv.Plane(center=(0.5, 0.5, 0.0), i_size=1.0, j_size=1.0)
    back_points = np.array([[0.5, 0.5, -1.0], [2.0, 2.0, -1.0]])
    back = pv.PolyData(back_points)

    picked_results = []
    pl = pv.Plotter(window_size=(400, 400))
    pl.add_mesh(plane, color='red')
    pl.add_mesh(back, color='blue', point_size=10)
    pl.enable_rectangle_point_picking(
        callback=picked_results.append, show=False, show_message=False
    )
    pl.view_xy()
    pl.show(auto_close=False, interactive=False)
    width, height = pl.window_size

    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width - 1, height - 1)

    picked = picked_results[0]
    assert isinstance(picked, pv.MultiBlock)
    assert isinstance(pl.picked_points, pv.MultiBlock)

    # One block per source mesh; each block keeps its own source IDs
    assert len(picked) == 2
    back_block = next(b for b in picked if np.allclose(b.points, back_points[1:2]))
    assert back_block.point_data['original_point_ids'].tolist() == [1]
    plane_block = next(b for b in picked if b.n_points == plane.n_points)
    np.testing.assert_array_equal(
        plane_block.point_data['original_point_ids'], np.arange(plane.n_points)
    )
    pl.close()


def test_rectangle_point_picking_empty_and_edge():
    """Empty and window-edge rectangles behave correctly."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = pv.PolyData(points)

    picked_results = []
    pl = pv.Plotter(window_size=(400, 400))
    pl.add_mesh(mesh, point_size=15)
    pl.enable_rectangle_point_picking(
        callback=picked_results.append, show=False, show_message=False
    )
    pl.view_xy()
    pl.show(auto_close=False, interactive=False)
    width, height = pl.window_size

    pl.iren._simulate_keypress('r')

    # Full-window rectangle uses the window edges as bounds
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width - 1, height - 1)
    assert pl.picked_points.n_points == 4

    # A rectangle in an empty area selects nothing and resets the picked state
    pl.iren._mouse_left_button_press(195, 195)
    pl.iren._mouse_left_button_release(205, 205)
    assert pl.picked_points is None
    assert len(picked_results) == 1  # callback is not invoked on empty selection

    # A zero-size rectangle away from any point also selects nothing
    pl.iren._mouse_left_button_press(200, 200)
    pl.iren._mouse_left_button_release(200, 200)
    assert pl.picked_points is None
    pl.close()


def test_rectangle_point_picking_subplot_attribution():
    """Rectangle point picking in subplots attributes points to the poked renderer."""
    left_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    right_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    picked_results = []
    pl = pv.Plotter(shape=(1, 2), window_size=(600, 400))
    pl.add_mesh(pv.PolyData(left_points), point_size=15)
    pl.enable_rectangle_point_picking(
        callback=picked_results.append, show=False, show_message=False
    )
    pl.subplot(0, 1)
    pl.add_mesh(pv.PolyData(right_points), point_size=15)

    pl.show(auto_close=False, interactive=False)
    width, height = pl.window_size
    half_width = width // 2

    pl.iren._simulate_keypress('r')

    # Drag covering the right subplot: only the right mesh is picked
    pl.iren._mouse_left_button_press(half_width, 0)
    pl.iren._mouse_left_button_release(width - 1, height - 1)
    right_picked = picked_results[-1]
    assert isinstance(right_picked, pv.PolyData)
    assert right_picked.n_points == 3
    np.testing.assert_array_equal(right_picked.point_data['original_point_ids'], np.arange(3))

    # Drag covering the left subplot: only the left mesh is picked
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(half_width - 1, height - 1)
    left_picked = picked_results[-1]
    assert isinstance(left_picked, pv.PolyData)
    assert left_picked.n_points == 2
    np.testing.assert_array_equal(left_picked.point_data['original_point_ids'], np.arange(2))
    pl.close()


def test_rectangle_point_picking_show_representation():
    """The selected-points representation is added, replaced, and cleared."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = pv.PolyData(points)

    pl = pv.Plotter(window_size=(400, 400))
    pl.add_mesh(mesh, point_size=15)
    pl.enable_rectangle_point_picking(show=True, show_message=False, color='blue', point_size=20)
    pl.view_xy()
    pl.show(auto_close=False, interactive=False)
    width, height = pl.window_size

    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width - 1, height - 1)
    assert pl.renderer.actors['_picked_points'] is not None

    # A second selection replaces rather than duplicates the representation
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width // 2, height // 2)
    assert sum(name == '_picked_points' for name in pl.renderer.actors) == 1

    # An empty selection clears the representation
    pl.iren._mouse_left_button_press(195, 195)
    pl.iren._mouse_left_button_release(205, 205)
    assert pl.picked_points is None
    assert '_picked_points' not in pl.renderer.actors
    pl.close()


@pytest.mark.parametrize(
    'mesh',
    [
        pv.PointSet([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        pv.ImageData(dimensions=(2, 2, 1)),
        pv.Cube().cast_to_unstructured_grid(),
    ],
)
def test_rectangle_point_picking_dataset_types(mesh):
    """Points are returned as PolyData for any pickable dataset type."""
    pl = pv.Plotter(window_size=(400, 400))
    pl.add_mesh(mesh)
    pl.enable_rectangle_point_picking(show=False, show_message=False)
    pl.show(auto_close=False, interactive=False)
    width, height = pl.window_size

    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width - 1, height - 1)

    picked = pl.picked_points
    assert isinstance(picked, pv.PolyData)
    assert 0 < picked.n_points <= mesh.n_points
    ids = picked.point_data['original_point_ids']
    assert np.all(ids >= 0)
    assert np.all(ids < mesh.n_points)
    np.testing.assert_allclose(picked.points, mesh.points[ids])
    pl.close()


def test_rectangle_point_picking_non_pickable_actor():
    """Non-pickable actors are excluded from the selection."""
    pickable = pv.PolyData(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    )
    non_pickable = pv.Sphere(center=(0.5, 0.5, 0.0))

    pl = pv.Plotter(window_size=(400, 400))
    pl.add_mesh(non_pickable, pickable=False, opacity=0.3)
    pl.add_mesh(pickable, point_size=15)
    pl.enable_rectangle_point_picking(show=False, show_message=False)
    pl.view_xy()
    pl.show(auto_close=False, interactive=False)
    width, height = pl.window_size

    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(0, 0)
    pl.iren._mouse_left_button_release(width - 1, height - 1)

    picked = pl.picked_points
    assert isinstance(picked, pv.PolyData)
    assert picked.n_points == pickable.n_points
    np.testing.assert_array_equal(
        picked.point_data['original_point_ids'], np.arange(pickable.n_points)
    )
    pl.close()


def test_rectangle_point_picking_composite_actor_skipped():
    """A composite-mapper actor is skipped without breaking other selections."""
    composite = pv.MultiBlock([pv.Cube(center=(10.0, 10.0, 10.0))])
    points = pv.PolyData(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # a swallowed callback traceback would fail here
        pl = pv.Plotter(window_size=(400, 400))
        pl.add_composite(composite)
        pl.add_mesh(points, point_size=15)
        pl.enable_rectangle_point_picking(show=False, show_message=False)
        pl.view_xy()
        pl.show(auto_close=False, interactive=False)
        width, height = pl.window_size

        pl.iren._simulate_keypress('r')
        pl.iren._mouse_left_button_press(0, 0)
        pl.iren._mouse_left_button_release(width - 1, height - 1)

    assert isinstance(pl.picked_points, pv.PolyData)
    assert pl.picked_points.n_points == points.n_points
    pl.close()


def test_rectangle_point_picking_forward():
    """The plotter method and attribute forward from the picking component."""
    pl = pv.Plotter()
    assert pl.picked_points is pl.picking.picked_points
    assert pl.picked_points is None
    pl.add_mesh(pv.Sphere())
    assert pl.enable_rectangle_point_picking(show_message=False) is None
    pl.close()


def test_cell_picking_interactive_subplot():
    n_cells = []

    def callback(picked_cells):
        n_cells.append(picked_cells.n_cells)

    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(pv.Sphere())  # TRIANGLE cells
    pl.enable_cell_picking(callback=callback)
    pl.subplot(0, 1)
    pl.add_mesh(pv.Box(level=4), show_edges=True)  # QUAD cells

    pl.show(auto_close=False, interactive=False)

    width, height = pl.window_size

    # Activate picking
    pl.iren._simulate_keypress('r')

    # select just the left-hand side
    pl.iren._mouse_left_button_press(width // 4, height // 2)
    pl.iren._mouse_left_button_release(width // 2, height)

    assert n_cells[0]
    assert pl.picked_cells
    assert pl.picked_cells.get_cell(0).type == pv.CellType.TRIANGLE

    # select just the right-hand side
    pl.iren._mouse_left_button_press(width - width // 4, height // 2)
    pl.iren._mouse_left_button_release(width, height)

    assert n_cells[0]
    assert pl.picked_cells
    assert pl.picked_cells.get_cell(0).type == pv.CellType.QUAD


@pytest.mark.parametrize('left_clicking', [False, True])
def test_point_picking(left_clicking):
    picked = []

    def callback(picked_point):
        picked.append(picked_point)

    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_point_picking(
        callback=callback,
        show_message=True,
        left_clicking=left_clicking,
    )
    # must show to activate the interactive renderer (for left_clicking)
    pl.show(auto_close=False)

    # simulate the pick
    width, height = pl.window_size

    if left_clicking:
        pl.iren._mouse_left_button_click(width // 2, height // 2)
    else:
        pl.iren._mouse_right_button_click(width // 2, height // 2)

    assert picked


@pytest.mark.skip_windows
@pytest.mark.parametrize('pickable_window', [False, True])
def test_point_picking_window(pickable_window):
    class Tracker:
        def __init__(self):
            self.last_picked = None

        def __call__(self, picked_point):
            self.last_picked = picked_point

    pl = pv.Plotter(
        window_size=(100, 100),
    )

    # bottom left corner, pickable
    sphere = pv.Sphere()
    sphere.translate([-1, -1, 0], inplace=True)
    pl.add_mesh(sphere, pickable=True)

    pl.camera_position = pv.CameraPosition(
        position=(0.0, 0.0, 8.5), focal_point=(0.0, 0.0, 0.0), viewup=(0.0, 1.0, 0.0)
    )

    tracker = Tracker()
    pl.enable_point_picking(
        callback=tracker,
        tolerance=0.2,
        pickable_window=pickable_window,
        picker='hardware',  # picker allows picking in the window
        # do not use point picker as it snaps to points
    )

    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker

    successful_pick = picker.Pick(25, 25, 0, renderer)
    assert successful_pick  # not a complete test
    assert tracker.last_picked is not None
    good_point = tracker.last_picked

    successful_pick = picker.Pick(75, 75, 0, renderer)
    assert not successful_pick  # not a complete test
    if pickable_window:
        assert not np.allclose(tracker.last_picked, good_point)  # make sure new point picked
    else:
        assert np.allclose(tracker.last_picked, good_point)  # make sure point did not change

    pl.close()


def test_path_picking():
    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_path_picking(
        show_message=True,
        callback=lambda path: None,  # noqa: ARG005
    )
    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker
    picker.Pick(50, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = pl.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    pl.close()


def test_geodesic_picking():
    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_geodesic_picking(
        show_message=True,
        callback=lambda path: None,  # noqa: ARG005
        show_path=True,
        keep_order=True,
    )
    pl.show(auto_close=False)

    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(45, 45, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = pl.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    pl.close()


def test_horizon_picking():
    sphere = pv.Sphere()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(sphere)
    pl.enable_horizon_picking(
        show_message=True,
        callback=lambda path: None,  # noqa: ARG005
        show_horizon=True,
    )
    # simulate the pick
    renderer = pl.renderer
    picker = pl.iren.picker
    # at least 3 picks
    picker.Pick(50, 50, 0, renderer)
    picker.Pick(49, 50, 0, renderer)
    picker.Pick(48, 50, 0, renderer)
    # pick nothing
    picker.Pick(0, 0, 0, renderer)
    # 'c' to clear
    clear_callback = pl.iren._key_press_event_callbacks['c']
    clear_callback[0]()
    pl.close()


@pytest.mark.usefixtures('verify_image_cache')
def test_fly_to_right_click(sphere):
    point = []

    def callback(click_point):
        point.append(click_point)

    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.enable_fly_to_right_click(callback=callback)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_right_button_click(width // 4, height // 2)

    # ensure callback was called and camera position changes due to "fly"
    assert cpos_before != pl.camera_position
    assert point
    pl.close()


@pytest.mark.usefixtures('verify_image_cache')
def test_fly_to_right_click_multi_render(sphere):
    """Same as enable as fly_to_right_click except with two renders for coverage"""
    point = []

    def callback(click_point):
        point.append(click_point)

    pl = pv.Plotter(shape=(1, 2))
    pl.add_mesh(sphere)
    pl.enable_fly_to_right_click(callback=callback)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_right_button_click(width // 8, height // 2)
    # ensure callback was called and camera position changes due to "fly"
    assert cpos_before != pl.camera_position
    assert point
    pl.close()


@pytest.mark.usefixtures('verify_image_cache')
def test_fly_to_mouse_position(sphere):
    """Same as enable as fly_to_right_click except with two renders for coverage"""
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show(auto_close=False)
    width, height = pl.window_size
    cpos_before = pl.camera_position
    pl.iren._mouse_move(width - width // 4, height // 2)
    pl.fly_to_mouse_position()
    assert cpos_before != pl.camera_position
    pl.close()


def test_block_picking(multiblock_poly):
    """Test we can pick a block."""
    pl = pv.Plotter()
    width, height = pl.window_size
    _actor, mapper = pl.add_composite(multiblock_poly)

    picked_blocks = []

    def turn_blue(index, dataset):  # noqa: ARG001
        mapper.block_attr[index].color = 'blue'
        picked_blocks.append(index)

    pl.enable_block_picking(callback=turn_blue)
    pl.show(auto_close=False)

    # click in the corner
    assert not picked_blocks
    pl.iren._mouse_left_button_click(0, 0)
    assert not picked_blocks

    # click directly in the middle
    pl.iren._mouse_left_button_click(width // 2, height // 2)
    assert mapper.block_attr[2].color

    assert pl.picked_block_index == picked_blocks[0]


@pytest.mark.parametrize('mode', ['mesh', 'cell', 'face', 'edge', 'point'])
def test_element_picking(mode):
    class Tracker:
        def __init__(self):
            self.last_picked = None

        def __call__(self, picked):
            self.last_picked = picked

    tracker = Tracker()

    mesh = pv.Wavelet()
    pl = pv.Plotter(
        window_size=(100, 100),
    )
    pl.add_mesh(mesh)
    pl.enable_element_picking(
        mode=mode,
        show_message=True,
        left_clicking=True,
        callback=tracker,
    )
    # must show to activate the interactive renderer (for left_clicking)
    pl.show(auto_close=False)

    # simulate the pick
    width, height = pl.window_size

    pl.iren._mouse_left_button_click(width // 2, height // 2)

    pl.close()

    assert tracker.last_picked is not None

    if mode == 'mesh':
        assert tracker.last_picked == mesh
    elif mode == 'cell':
        assert tracker.last_picked.n_points == 8
    elif mode == 'face':
        assert tracker.last_picked.n_points == 4
    elif mode == 'edge':
        assert tracker.last_picked.n_points == 2
    elif mode == 'point':
        assert isinstance(tracker.last_picked, pv.PolyData)
        assert tracker.last_picked.n_points == 1


def test_element_picking_point_preserves_data():
    """Test that point picking with element handler preserves point data."""

    class Tracker:
        def __init__(self):
            self.last_picked = None

        def __call__(self, picked):
            self.last_picked = picked

    tracker = Tracker()

    # Create a mesh with point data
    mesh = pv.Sphere()
    mesh.point_data['test_data'] = np.arange(mesh.n_points)
    mesh.point_data['test_scalars'] = mesh.points[:, 0]  # x-coordinates

    pl = pv.Plotter(window_size=(100, 100))
    pl.add_mesh(mesh)
    pl.enable_element_picking(
        mode='point',
        show_message=True,
        left_clicking=True,
        callback=tracker,
    )
    pl.show(auto_close=False)

    # Simulate the pick
    width, height = pl.window_size
    pl.iren._mouse_left_button_click(width // 2, height // 2)

    pl.close()

    # Verify the picked point exists and is a PolyData
    assert tracker.last_picked is not None
    assert isinstance(tracker.last_picked, pv.PolyData)
    assert tracker.last_picked.n_points == 1

    # Verify point data is preserved
    assert 'test_data' in tracker.last_picked.point_data
    assert 'test_scalars' in tracker.last_picked.point_data
    assert tracker.last_picked.point_data['test_data'].size == 1
    assert tracker.last_picked.point_data['test_scalars'].size == 1

    # Verify vtkOriginalPointIds is present
    assert 'vtkOriginalPointIds' in tracker.last_picked.point_data
    original_id = tracker.last_picked.point_data['vtkOriginalPointIds'][0]
    assert 0 <= original_id < mesh.n_points

    # Verify the point data values match the original mesh
    assert (
        tracker.last_picked.point_data['test_data'][0] == mesh.point_data['test_data'][original_id]
    )
    assert np.isclose(
        tracker.last_picked.point_data['test_scalars'][0],
        mesh.point_data['test_scalars'][original_id],
    )


def test_switch_picking_type():
    pl = pv.Plotter()
    width, height = pl.window_size
    pl.add_mesh(pv.Sphere())

    cells = []

    def callback(picked):
        cells.append(picked)

    pl.enable_cell_picking(callback=callback)
    with pytest.raises(PyVistaPickingError):
        pl.enable_point_picking()

    pl.show(auto_close=False, interactive=False)
    pl.iren._simulate_keypress('r')
    pl.iren._mouse_left_button_press(width // 4, height // 4)
    pl.iren._mouse_left_button_release(width, height)

    assert cells
    assert isinstance(cells[0], pv.UnstructuredGrid)
    assert pl.picked_cells is not None

    # Now switch to point picking
    pl.disable_picking()

    points = []

    def callback(click_point):
        points.append(click_point)

    pl.enable_point_picking(callback=callback)
    # simulate the pick
    width, height = pl.window_size

    pl.iren._mouse_right_button_click(width // 3, height // 2)

    pl.close()

    assert points
    assert len(points[0]) == 3


@pytest.mark.parametrize('picker', ['foo', 1000])
def test_picker_raises(picker, mocker: MockerFixture):
    pl = pv.Plotter()  # patching need to occur after init

    from pyvista.plotting import picking

    m = mocker.patch.object(typ := picking.PickerType, 'from_any')
    m.return_value = None

    types = [typ.POINT, typ.CELL, typ.HARDWARE, typ.VOLUME]
    match = re.escape(
        f'Invalid picker choice for surface picking. Use one of: {types}',
    )
    with pytest.raises(ValueError, match=match):
        pl.enable_surface_point_picking(picker=picker)

    m.assert_called_once_with(picker)


def test_block_picking_across_four_subplots():
    """Test block picking on a 2x2 subplot layout with a MultiBlock of two spheres."""
    sphere1 = pv.Sphere(center=(-1.0, 0.0, 0.0), radius=0.5)
    sphere2 = pv.Sphere(center=(1.0, 0.0, 0.0), radius=0.5)
    blocks = pv.MultiBlock([sphere1, sphere2])

    # Record each block index picked up
    picked = []

    def callback(index, dataset):  # noqa: ARG001
        picked.append(index)

    pl = pv.Plotter(shape=(2, 2), window_size=[400, 400])
    for row in range(2):
        for col in range(2):
            pl.subplot(row, col)
            pl.add_composite(blocks, color='w', pickable=True)

    pl.enable_block_picking(callback, side='left')

    # Using camera_position to ensure consistent view for testing
    camera_position = pv.CameraPosition(
        position=(3.0, 3.0, 3.0),
        focal_point=(0.0, 0.0, 0.0),
        viewup=(0.0, 0.0, 1.0),
    )

    pl.show(
        auto_close=False,
        cpos=camera_position,
    )

    # Using hard-coded click positions to ensure consistent testing
    click_coords = [
        [45, 266],
        [147, 326],
        [245, 263],
        [352, 321],
        [44, 59],
        [147, 121],
        [243, 60],
        [349, 125],
    ]

    for x, y in click_coords:
        pl.iren._mouse_left_button_click(x, y)
    pl.close()

    expected = [2, 1] * 4
    assert picked == expected, f'Picked indices {picked}, but expected {expected}'


def test_pick_click_position_forward(sphere):
    """``Plotter.pick_click_position`` triggers a click-position store and pick."""
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show(auto_close=False)
    pos = pl.pick_click_position()
    assert isinstance(pos, tuple)
    assert len(pos) == 3
    pl.close()


def test_pick_mouse_position_forward(sphere):
    """``Plotter.pick_mouse_position`` triggers a mouse-position store and pick."""
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.show(auto_close=False)
    pos = pl.pick_mouse_position()
    assert isinstance(pos, tuple)
    assert len(pos) == 3
    pl.close()


def test_picked_path_geodesic_horizon_forward():
    """The deprecated picked-path/geodesic/horizon getters forward from the component."""
    pl = pv.Plotter()
    assert pl.picked_path is pl.picking.picked_path
    assert pl.picked_geodesic is pl.picking.picked_geodesic
    assert pl.picked_horizon is pl.picking.picked_horizon
    pl.close()


def test_left_click_picking_observer_branch(sphere):
    """``enable_point_picking(left_clicking=True)`` exercises the left-click branch."""
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    pl.enable_point_picking(left_clicking=True, callback=lambda p: None)  # noqa: ARG005
    assert pl.picking._picking_left_clicking_observer is not None
    pl.disable_picking()
    assert pl.picking._picking_left_clicking_observer is None
    pl.close()
