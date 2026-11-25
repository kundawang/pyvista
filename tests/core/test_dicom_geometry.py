"""Tests for DICOM patient geometry (origin and direction matrix)."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.uid import generate_uid
import pytest

import pyvista as pv

if TYPE_CHECKING:
    from pathlib import Path


def _write_dicom(
    path: Path,
    pixels: np.ndarray,
    *,
    iop=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ipp=(0.0, 0.0, 0.0),
    pixel_spacing=(1.0, 1.0),
    instance_number=1,
    series_uid=None,
) -> Path:
    rows, columns = pixels.shape
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = series_uid or generate_uid()
    dataset.Modality = 'CT'
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = 'MONOCHROME2'
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    dataset.ImageOrientationPatient = [float(v) for v in iop]
    dataset.ImagePositionPatient = [float(v) for v in ipp]
    dataset.InstanceNumber = int(instance_number)
    dataset.PixelData = pixels.astype(np.int16).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def rotated_geometry():
    theta = np.arctan2(4.0, 3.0)  # cos=0.6, sin=0.8: off-axis but exact-ish
    row_cosines = np.array([np.cos(theta), np.sin(theta), 0.0])
    column_cosines = np.array([-np.sin(theta), np.cos(theta), 0.0])
    normal = np.cross(row_cosines, column_cosines)
    ipp = np.array([10.0, 20.0, 30.0])
    row_spacing, column_spacing = 2.0, 3.0
    return row_cosines, column_cosines, normal, ipp, row_spacing, column_spacing


def _expected_world(geometry, row, column, slice_index=0):
    x, y, normal, ipp, row_spacing, column_spacing = geometry
    slice_spacing = 4.0
    return (
        ipp
        + column * column_spacing * x
        + row * row_spacing * y
        + slice_index * slice_spacing * normal
    )


def test_dicom_rotated_single_pixel_geometry_and_array(tmp_path, rotated_geometry):
    x, y, normal, ipp, row_spacing, column_spacing = rotated_geometry
    rows, columns = 5, 6
    pixels = np.array(
        [[r * 100 + c + 1 for c in range(columns)] for r in range(rows)], dtype=np.int16
    )
    path = _write_dicom(
        tmp_path / 'rotated.dcm',
        pixels,
        iop=(*x, *y),
        ipp=tuple(ipp),
        pixel_spacing=(row_spacing, column_spacing),
    )

    image = pv.DICOMReader(str(path)).read()

    np.testing.assert_allclose(image.origin, ipp, atol=1e-5)
    np.testing.assert_allclose(image.direction_matrix, np.column_stack((x, y, normal)), atol=1e-5)
    assert image.spacing[0] == pytest.approx(column_spacing)
    assert image.spacing[1] == pytest.approx(row_spacing)

    # Read-back pixel array matches DICOM row/column order.
    np.testing.assert_array_equal(
        np.asarray(image.active_scalars).reshape(rows, columns),
        pydicom.dcmread(path).pixel_array,
    )

    # Every designated pixel is at its DICOM world position.
    matrix = image.index_to_physical_matrix
    for r in range(rows):
        for c in range(columns):
            world = matrix @ np.array([c, r, 0, 1.0])
            np.testing.assert_allclose(
                world[:3], _expected_world(rotated_geometry, r, c), atol=1e-3
            )


def test_dicom_rotated_stack_order_and_geometry(tmp_path, rotated_geometry):
    x, y, normal, ipp, row_spacing, column_spacing = rotated_geometry
    rows, columns, n_slices = 4, 5, 4
    slice_spacing = 4.0
    series_uid = generate_uid()
    stack = np.empty((n_slices, rows, columns), dtype=np.int16)
    directory = tmp_path / 'stack'
    directory.mkdir()
    for k in range(n_slices):
        pixels = np.array(
            [[r * 10 + c + 1 + k * 1000 for c in range(columns)] for r in range(rows)],
            dtype=np.int16,
        )
        stack[k] = pixels
        _write_dicom(
            directory / f'IM_{k:04d}.dcm',
            pixels,
            iop=(*x, *y),
            ipp=tuple(ipp + k * slice_spacing * normal),
            pixel_spacing=(row_spacing, column_spacing),
            instance_number=k + 1,
            series_uid=series_uid,
        )

    image = pv.DICOMReader(str(directory)).read()

    assert image.dimensions == (columns, rows, n_slices)
    np.testing.assert_allclose(image.origin, ipp, atol=1e-4)
    np.testing.assert_allclose(image.direction_matrix, np.column_stack((x, y, normal)), atol=1e-5)
    assert image.spacing[2] == pytest.approx(slice_spacing)
    np.testing.assert_array_equal(
        np.asarray(image.active_scalars).reshape(n_slices, rows, columns), stack
    )
    for k in range(n_slices):
        world = image.index_to_physical_matrix @ np.array([0, 0, k, 1.0])
        np.testing.assert_allclose(
            world[:3], _expected_world(rotated_geometry, 0, 0, k), atol=1e-3
        )


def test_dicom_axis_aligned_default_geometry_unchanged(tmp_path):
    rows, columns = 4, 5
    pixels = np.arange(rows * columns, dtype=np.int16).reshape(rows, columns) + 1
    path = _write_dicom(tmp_path / 'axis.dcm', pixels)

    image = pv.DICOMReader(str(path)).read()

    np.testing.assert_allclose(image.origin, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(image.direction_matrix, np.eye(3))
    np.testing.assert_array_equal(np.asarray(image.active_scalars).reshape(rows, columns), pixels)


def test_dicom_missing_geometry_warns_and_defaults(tmp_path):
    pixels = np.arange(6, dtype=np.int16).reshape(2, 3)
    path = _write_dicom(tmp_path / 'missing.dcm', pixels)
    dataset = pydicom.dcmread(path)
    del dataset.ImageOrientationPatient
    del dataset.ImagePositionPatient
    dataset.save_as(path)

    with pytest.warns(UserWarning, match='ImageOrientationPatient'):
        image = pv.DICOMReader(str(path)).read()

    np.testing.assert_allclose(image.origin, 0.0)
    np.testing.assert_allclose(image.direction_matrix, np.eye(3))


def test_dicom_contradictory_orientation_warns_and_defaults(tmp_path):
    pixels = np.arange(6, dtype=np.int16).reshape(2, 3)
    path = _write_dicom(tmp_path / 'contradictory.dcm', pixels)
    dataset = pydicom.dcmread(path)
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.5, 0.5, 0.0]
    dataset.save_as(path)

    with pytest.warns(UserWarning, match='ImageOrientationPatient'):
        image = pv.DICOMReader(str(path)).read()

    np.testing.assert_allclose(image.direction_matrix, np.eye(3))


def test_dicom_scout_from_other_series_does_not_hijack_origin(tmp_path, rotated_geometry):
    from pyvista.core.utilities import reader as reader_module

    x, y, normal, ipp, row_spacing, column_spacing = rotated_geometry
    rows, columns, n_slices, slice_spacing = 4, 5, 3, 4.0
    directory = tmp_path / 'with_scout'
    directory.mkdir()
    main_uid, scout_uid = generate_uid(), generate_uid()
    for k in range(n_slices):
        _write_dicom(
            directory / f'IM_{k}.dcm',
            np.full((rows, columns), k + 1, dtype=np.int16),
            iop=(*x, *y),
            ipp=tuple(ipp + k * slice_spacing * normal),
            pixel_spacing=(row_spacing, column_spacing),
            instance_number=k + 1,
            series_uid=main_uid,
        )
    scout_ipp = ipp - 50.0 * normal
    _write_dicom(
        directory / 'SCOUT.dcm',
        np.full((rows, columns), 9, dtype=np.int16),
        iop=(*x, *y),
        ipp=tuple(scout_ipp),
        pixel_spacing=(row_spacing, column_spacing),
        instance_number=99,
        series_uid=scout_uid,
    )

    # Given VTK stacked the three-slice series, geometry must come from that series.
    origin, direction = reader_module._read_dicom_patient_geometry(str(directory), n_slices)
    np.testing.assert_allclose(origin, ipp, atol=1e-3)
    np.testing.assert_allclose(direction, np.column_stack((x, y, normal)), atol=1e-5)

    # If VTK instead exposed the one-slice scout, its own origin is used, never the stack's.
    scout_geometry = reader_module._read_dicom_patient_geometry(str(directory), 1)
    np.testing.assert_allclose(scout_geometry[0], scout_ipp, atol=1e-3)


def test_dicom_gantry_tilt_warns_and_defaults(tmp_path, rotated_geometry):
    x, y, normal, ipp, _, _ = rotated_geometry
    n_slices, slice_spacing = 3, 4.0
    directory = tmp_path / 'tilted'
    directory.mkdir()
    series_uid = generate_uid()
    for k in range(n_slices):
        tilted_ipp = ipp + k * (slice_spacing * normal + 2.0 * x)
        _write_dicom(
            directory / f'T_{k}.dcm',
            np.full((4, 5), k + 1, dtype=np.int16),
            iop=(*x, *y),
            ipp=tuple(tilted_ipp),
            instance_number=k + 1,
            series_uid=series_uid,
        )

    with pytest.warns(UserWarning, match='ImageOrientationPatient'):
        image = pv.DICOMReader(str(directory)).read()

    np.testing.assert_allclose(image.direction_matrix, np.eye(3))


def test_dicom_without_pydicom_uses_safe_default(monkeypatch, tmp_path):
    path = _write_dicom(tmp_path / 'x.dcm', np.arange(6, dtype=np.int16).reshape(2, 3))
    real_find_spec = importlib.util.find_spec

    def missing_pydicom(name, package=None):
        if name == 'pydicom':
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, 'find_spec', missing_pydicom)

    with pytest.warns(UserWarning, match='pydicom'):
        image = pv.DICOMReader(str(path)).read()

    np.testing.assert_allclose(image.origin, 0.0)
    np.testing.assert_allclose(image.direction_matrix, np.eye(3))
