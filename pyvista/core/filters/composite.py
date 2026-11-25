"""Filters module with a class to manage filters/algorithms for composite datasets."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista._version import _is_deprecation_due
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.filters import _apply_points_dtype
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_object import DataObjectFilters
from pyvista.core.filters.data_object import _make_reference_volume
from pyvista.core.filters.data_object import _validate_reference_volume_options
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.utilities._cell_lengths import _cell_length_percentile
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from pyvista import ImageData
    from pyvista import MultiBlock
    from pyvista import PolyData
    from pyvista import UnstructuredGrid
    from pyvista.core._typing_core import VectorLikeFloat
    from pyvista.core._typing_core import VectorLikeInt
    from pyvista.core._typing_core import _DataSetType
    from pyvista.core.composite import _TypeMultiBlockLeaf


@abstract_class
class CompositeFilters(DataObjectFilters):
    """An internal class to manage filters/algorithms for composite datasets."""

    def generic_filter(  # type:ignore[misc]
        self: MultiBlock[Any],
        function: str | Callable[..., _TypeMultiBlockLeaf],
        /,
        *args,
        **kwargs,
    ) -> MultiBlock:
        """Apply any filter to all nested blocks recursively.

        This filter applies a user-specified function or method to all blocks in
        this :class:`~pyvista.MultiBlock`.

        .. note::

            If an ``inplace`` keyword is used, this ``MultiBlock`` is modified
            in-place along with all blocks.

        .. note::

            By default, the specified ``function`` is not applied to any ``None``
            blocks. These are simply skipped and passed through to the output.

            For advanced use, it is possible to apply the filter to ``None`` blocks
            by using the undocumented keyword ``_skip_none=False``.

        .. versionadded:: 0.45

        Parameters
        ----------
        function : Callable | str
            Callable function or name of the method to apply to each block. The function
            should accept a :class:`~pyvista.DataSet` as input and return either a
            :class:`~pyvista.DataSet` or :class:`~pyvista.MultiBlock` as output.

        *args : Any, optional
            Arguments to use with the specified ``function``.

        **kwargs : Any, optional
            Keyword arguments to use with the specified ``function``.

        Returns
        -------
        MultiBlock
            Filtered dataset.

        Raises
        ------
        Exception
            The error a block raises is re-raised as is, with the index, name, and
            type of that block added to its message.

            .. versionchanged:: 0.50

                The error keeps its own class. Previously every error was replaced by
                a ``RuntimeError``.

        See Also
        --------
        pyvista.MultiBlock.flatten
        pyvista.MultiBlock.recursive_iterator
        pyvista.MultiBlock.clean

        Examples
        --------
        Create a :class:`~pyvista.MultiBlock` with various mesh types.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> import numpy as np
        >>> volume = examples.load_uniform()
        >>> poly = examples.load_ant()
        >>> unstructured = examples.load_tetbeam()
        >>> multi = pv.MultiBlock([volume, poly, unstructured])

        >>> [type(block) for block in multi]  # doctest: +NORMALIZE_WHITESPACE
        [<class 'pyvista.core.grid.ImageData'>,
         <class 'pyvista.core.pointset.PolyData'>,
         <class 'pyvista.core.pointset.UnstructuredGrid'>]

        Use the generic filter to apply :meth:`~pyvista.DataSet.cast_to_unstructured_grid`
        to all blocks.

        >>> filtered = multi.generic_filter('cast_to_unstructured_grid')

        >>> [type(block) for block in filtered]  # doctest: +NORMALIZE_WHITESPACE
        [<class 'pyvista.core.pointset.UnstructuredGrid'>,
         <class 'pyvista.core.pointset.UnstructuredGrid'>,
         <class 'pyvista.core.pointset.UnstructuredGrid'>]

        Use the :meth:`~pyvista.DataSetFilters.partition` filter on all blocks.
        Any arguments can be specified as though the filter is being used directly.

        >>> filtered = multi.generic_filter('partition', 4, as_composite=True)

        Any function can be used as long as it returns a :class:`~pyvista.DataSet` or
        :class:`~pyvista.MultiBlock`. For example, we can normalize each block
        independently to have bounds between ``-0.5`` and ``0.5``.

        >>> def normalize_bounds(dataset):
        ...     # Center the dataset
        ...     dataset = dataset.translate(-np.array(dataset.center))
        ...     # Scale the dataset
        ...     factor = 1 / np.array(dataset.bounds_size)
        ...     return dataset.scale(factor)

        >>> filtered = multi.generic_filter(normalize_bounds)
        >>> filtered
        MultiBlock[ImageData | PolyData | UnstructuredGrid] (...)
          N Blocks:   3
          X Bounds:   -5.000e-01, 5.000e-01
          Y Bounds:   -5.000e-01, 5.000e-01
          Z Bounds:   -5.000e-01, 5.000e-01

        The generic filter will fail if the filter can only be applied to some blocks
        but not others. For example, it is not possible to use the
        :meth:`~pyvista.ImageDataFilters.resample` filter generically since the
        ``MultiBlock`` above is heterogeneous and contains some blocks which are not
        :class:`~pyvista.ImageData`.

        >>> multi.generic_filter('resample', 0.5)  # doctest:+SKIP
        AttributeError: The filter 'resample'
        could not be applied to the block at index 1 with name 'Block-01' and type PolyData:
        'PolyData' object has no attribute 'resample'

        Use a custom function instead to apply the generic filter conditionally. Here we
        filter the image blocks but simply pass-through a copy of any other blocks.

        >>> def conditional_resample(dataset, *args, **kwargs):
        ...     if isinstance(dataset, pv.ImageData):
        ...         return dataset.resample(*args, **kwargs)
        ...     return dataset.copy()

        >>> filtered = multi.generic_filter(conditional_resample, 0.5)

        """
        # Set default undocumented kwargs. A function is used here to prevent IDEs from
        # suggesting these keywords to users.

        def get_iterator_kwargs(kwargs_: dict[str, Any]) -> tuple[bool, bool]:
            # Skip None blocks by default
            skip_none_: bool = kwargs_.pop('_skip_none', True)
            # Do not skip empty blocks by default
            skip_empty_: bool = kwargs_.pop('_skip_empty', False)
            return skip_none_, skip_empty_

        skip_none, skip_empty = get_iterator_kwargs(kwargs)

        def apply_filter(  # noqa: PLR0917
            function_: str | Callable[..., _TypeMultiBlockLeaf],
            ids_: tuple[int, ...],
            name_: str,
            block_: _TypeMultiBlockLeaf,
        ) -> _TypeMultiBlockLeaf:
            try:
                function_ = (
                    getattr(block_, function_)
                    if isinstance(function_, str)
                    else functools.partial(function_, block_)
                )
                output_ = function_(**kwargs) if len(args) == 0 else function_(*args, **kwargs)
            except Exception as e:
                # Name the block in the error, keeping the error's own class
                func_name = (
                    function_.func if isinstance(function_, functools.partial) else function_
                )
                obj_name = type(block).__name__
                index: int | str
                if len(ids_) == 1:
                    index = ids_[0]
                    nested = ' '
                else:
                    nested = ' nested '
                    index = _format_nested_index(ids)
                msg = (
                    f"The filter '{func_name}'\n"
                    f'could not be applied to the{nested}block at index {index} with '
                    f"name '{name_}' and type {obj_name}"
                )
                e.args = (f'{msg}:\n{e}' if str(e) else f'{msg}.', *e.args[1:])
                raise
            return output_

        def get_iterator(
            multi: MultiBlock[Any], *, skip_none_: bool, skip_empty_: bool
        ) -> Iterator[tuple[tuple[int, ...], str, _TypeMultiBlockLeaf]]:
            # `nested_ids` makes every id a tuple, which the overloads cannot prove here
            return multi.recursive_iterator(  # type: ignore[return-value]
                'all', skip_none=skip_none_, skip_empty=skip_empty_, nested_ids=True
            )

        # Apply filter in-place
        inplace = kwargs.get('inplace')
        if inplace:
            for ids, name, block in get_iterator(
                self, skip_none_=skip_none, skip_empty_=skip_empty
            ):
                apply_filter(function, ids, name, block)
            return self

        # Create a copy and replace all the blocks
        output = pv.MultiBlock()
        output.shallow_copy(self, recursive=True)
        for ids, name, block in get_iterator(output, skip_none_=skip_none, skip_empty_=skip_empty):
            filtered = apply_filter(function, ids, name, block)
            # Only replace if necessary
            if filtered is not block:
                output.replace(ids, filtered)
        return output

    def extract_geometry(self) -> PolyData:
        """Extract the surface the geometry of all blocks.

        Place this filter at the end of a pipeline before a polydata
        consumer such as a polydata mapper to extract geometry from
        all blocks and append them to one polydata object.

        Returns
        -------
        pyvista.PolyData
            Surface of the composite dataset.

        """
        msg = '`extract_geometry` is deprecated. Use `extract_surface(algorithm=None)` instead.'
        warn_external(msg, PyVistaDeprecationWarning)
        if _is_deprecation_due((0, 50)):  # pragma: no cover
            msg = 'Convert this deprecation warning into an error.'
            raise RuntimeError(msg)
        if pv.version_info >= (0, 51):  # pragma: no cover
            msg = 'Remove this deprecated filter.'
            raise RuntimeError(msg)
        return self._composite_geometry_filter()

    def _composite_geometry_filter(self) -> PolyData:
        gf = _vtk.vtkCompositeDataGeometryFilter()
        gf.SetInputData(self)
        _update_alg(gf)
        return _apply_points_dtype(wrap(gf.GetOutputDataObject(0)), algorithm=gf)

    def combine(  # type: ignore[misc]
        self: MultiBlock[Any], *, merge_points: bool = False, tolerance: float = 0.0
    ) -> UnstructuredGrid:
        """Combine all blocks into a single unstructured grid.

        Parameters
        ----------
        merge_points : bool, default: False
            Merge coincident points. The default differs from
            :func:`pyvista.merge`, which merges points by default.

            .. versionchanged:: 0.50
                Points of nested blocks are merged in a single pass, so a nonzero
                ``tolerance`` gives the same result as for the equivalent flat
                :class:`~pyvista.MultiBlock`.

        tolerance : float, default: 0.0
            The absolute tolerance to use to find coincident points when
            ``merge_points=True``.

        Returns
        -------
        pyvista.UnstructuredGrid
            Combined blocks.

        Examples
        --------
        Combine blocks within a multiblock without merging points.

        >>> import pyvista as pv
        >>> block = pv.MultiBlock(
        ...     [
        ...         pv.Cube(clean=False),
        ...         pv.Cube(center=(1, 0, 0), clean=False),
        ...     ]
        ... )
        >>> merged = block.combine()
        >>> merged.n_points
        48

        Combine blocks and merge points

        >>> merged = block.combine(merge_points=True)
        >>> merged.n_points
        12

        """
        alg = _vtk.vtkAppendFilter()
        n_inputs = 0
        for block in self.recursive_iterator(skip_none=True):
            alg.AddInputData(block)
            n_inputs += 1
        if n_inputs == 0:
            # Nothing to append, so give back an empty grid
            return pv.UnstructuredGrid()
        alg.SetMergePoints(merge_points)
        alg.SetTolerance(tolerance)
        _update_alg(alg)
        output: _vtk.vtkUnstructuredGrid = alg.GetOutputDataObject(0)
        return _apply_points_dtype(wrap(output), algorithm=alg)

    def voxelize_labels(  # type: ignore[misc]
        self: MultiBlock[Any],
        overlap: str = 'last',
        *,
        reference_volume: ImageData | None = None,
        dimensions: VectorLikeInt | None = None,
        spacing: float | VectorLikeFloat | None = None,
        target_n_points: int | None = None,
        max_n_points: int | None = None,
        rounding_func: Callable[[VectorLikeFloat], VectorLikeInt] | None = None,
        cell_length_percentile: float | None = None,
        cell_length_sample_size: int | None = None,
        origin: VectorLikeFloat | None = None,
        labels_name: str = 'labels',
        progress_bar: bool = False,
        **kwargs: Any,
    ) -> ImageData:
        """Voxelize the blocks into one image with an integer label per block.

        Every block is rasterized into the same grid, so all blocks share one
        origin, spacing, and extent. Background points are labelled ``0`` and
        each leaf block receives a stable, nonzero label equal to its 1-based
        position in the depth-first traversal of the blocks, including
        ``None`` and empty leaf slots. Reordering the blocks therefore
        relabels them by their new positions in a predictable way, and empty
        blocks contribute no foreground points. Nested blocks are traversed
        recursively.

        Where the closed surfaces of two blocks overlap, only one label can
        own the point. The ``overlap`` parameter selects whether the first or
        last block in traversal order wins.

        The geometry options behave exactly like
        :meth:`pyvista.DataSetFilters.voxelize_binary_mask`: the bounds of all
        non-empty blocks are combined, and the same spacing and dimension
        rules apply. Use ``reference_volume`` for full and exact control of
        the output grid.

        .. versionadded:: 0.50.0

        Parameters
        ----------
        overlap : {'last', 'first', 'zero'}, default: 'last'
            Rule for points covered by more than one block:

            * ``'last'``: the last block in traversal order wins.
            * ``'first'``: the first block in traversal order wins.
            * ``'zero'``: multiply covered points are reset to background
              ``0``.

        reference_volume : ImageData, optional
            Volume whose ``dimensions``, ``origin``, ``spacing``, ``offset``,
            and ``direction_matrix`` define the output geometry exclusively.

        dimensions : VectorLikeInt, optional
            Dimensions (number of points along each axis) of the output image.

        spacing : float | VectorLikeFloat, optional
            Approximate spacing used to size the output image.

        target_n_points : int, optional
            Approximate number of output points to aim for with isotropic
            spacing.

        max_n_points : int, optional
            Strict upper bound on the number of output points.

        rounding_func : Callable, optional
            Rounding function used when deriving dimensions from spacing.

        cell_length_percentile : float, optional
            Percentile used to estimate the default spacing from the cells of
            all blocks. Default is ``0.1``.

        cell_length_sample_size : int, optional
            Number of cells sampled to estimate the default spacing.

        origin : VectorLikeFloat, optional
            Explicit origin of the output image. Overrides the origin derived
            from the combined bounds. Prefer ``reference_volume`` to control
            the geometry fully.

        labels_name : str, default: ``'labels'``
            Name of the output point-data label array. The same label values
            are also stored under the ``'mask'`` key for compatibility with
            :meth:`~pyvista.DataSetFilters.voxelize_binary_mask`.

        progress_bar : bool, default: False
            Display a progress bar while rasterizing each block.

        **kwargs : Any, optional
            Additional keyword arguments accepted for compatibility:
            ``on_overlap`` (alias of ``overlap``), ``array_name``,
            ``label_name`` and ``name`` (aliases of ``labels_name``),
            ``extent`` (VTK extent used to build a ``reference_volume``), and
            ``label``/``labels`` (boolean flags are ignored; an integer
            sequence sets explicit label values per leaf slot).

        Returns
        -------
        ImageData
            Image with point-data label array ``labels_name`` (default
            ``'labels'``) and a duplicate ``'mask'`` array holding the same
            label values. Labels are ``1`` onward and ``0`` is background.

        See Also
        --------
        pyvista.DataSetFilters.voxelize_binary_mask
            Single-mesh binary mask this filter applies to each block.

        Examples
        --------
        Label three unit-scale cubes on one shared grid. The first two cubes
        are disjoint and the third overlaps both of them. Sample points sit at
        the half-integer coordinates of the grid, so each label can be checked
        directly against the cube bounds.

        >>> import pyvista as pv
        >>> bounds = [
        ...     (0, 2, 0, 2, 0, 2),
        ...     (4, 6, 0, 2, 0, 2),
        ...     (1, 5, 0, 2, 0, 2),
        ... ]
        >>> blocks = pv.MultiBlock([pv.Box(bounds=b) for b in bounds])
        >>> reference_volume = pv.ImageData(
        ...     dimensions=(7, 2, 2), spacing=(1, 1, 1), origin=(0.5, 0.5, 0.5)
        ... )
        >>> labels = blocks.voxelize_labels(reference_volume=reference_volume)
        >>> labels['labels'].reshape(2, 2, 7)[0, 0].tolist()
        [1, 3, 3, 3, 3, 2, 0]

        With ``overlap='first'`` the first two blocks keep their overlap
        points instead of the third.

        >>> labels_first = blocks.voxelize_labels(
        ...     overlap='first', reference_volume=reference_volume
        ... )
        >>> labels_first['labels'].reshape(2, 2, 7)[0, 0].tolist()
        [1, 1, 3, 3, 2, 2, 0]

        """
        extent = kwargs.pop('extent', None)
        overlap, labels_name, explicit_labels = _parse_label_kwargs(overlap, labels_name, kwargs)
        mode = _overlap_mode(overlap)

        if extent is not None and reference_volume is None:
            reference_volume = pv.ImageData()
            reference_volume.extent = extent
            if spacing is not None:
                reference_volume.spacing = _broadcast3(spacing)
                spacing = None
            if origin is not None:
                reference_volume.origin = _broadcast3(origin)
                origin = None

        leaves = list(self.recursive_iterator('blocks', skip_none=False))
        non_empty = [
            block
            for block in leaves
            if block is not None and not (block.n_points == 0 and block.n_cells == 0)
        ]
        if explicit_labels is not None and len(explicit_labels) not in (
            len(leaves),
            len(non_empty),
        ):
            msg = (
                f'`labels` has {len(explicit_labels)} values but the MultiBlock has '
                f'{len(leaves)} leaf slots and {len(non_empty)} non-empty blocks.'
            )
            raise ValueError(msg)

        if not non_empty:
            return self._empty_labels_volume(
                reference_volume=reference_volume,
                dimensions=dimensions,
                spacing=spacing,
                origin=origin,
                labels_name=labels_name,
                n_slots=len(leaves),
            )

        if reference_volume is not None:
            _validate_reference_volume_options(
                reference_volume=reference_volume,
                dimensions=dimensions,
                spacing=spacing,
                target_n_points=target_n_points,
                max_n_points=max_n_points,
                rounding_func=rounding_func,
                cell_length_percentile=cell_length_percentile,
                cell_length_sample_size=cell_length_sample_size,
            )
            volume = _make_reference_volume(
                self,
                reference_volume=reference_volume,
                dimensions=dimensions,
                spacing=spacing,
                target_n_points=target_n_points,
                max_n_points=max_n_points,
                rounding_func=rounding_func,
                cell_length_percentile=cell_length_percentile,
                cell_length_sample_size=cell_length_sample_size,
            )
        elif dimensions is not None and spacing is not None:
            # Dimensions and spacing together define an exact VTK grid rather
            # than the approximate spacing the single-mesh estimator derives.
            volume = pv.ImageData()
            volume.dimensions = dimensions
            volume.spacing = _broadcast3(spacing)
            volume.origin = (
                _broadcast3(origin) if origin is not None else np.array(self.bounds[::2])
            )
        else:
            _validate_reference_volume_options(
                reference_volume=reference_volume,
                dimensions=dimensions,
                spacing=spacing,
                target_n_points=target_n_points,
                max_n_points=max_n_points,
                rounding_func=rounding_func,
                cell_length_percentile=cell_length_percentile,
                cell_length_sample_size=cell_length_sample_size,
            )
            if dimensions is None and spacing is None and target_n_points is None:
                # Estimate the default spacing from every block's cells, matching
                # the single-mesh default of the tenth percentile.
                if cell_length_percentile is None:
                    cell_length_percentile = 0.1
                if cell_length_sample_size is None:
                    cell_length_sample_size = 100_000
                spacing = _multiblock_default_spacing(
                    non_empty, cell_length_percentile, cell_length_sample_size
                )
                if spacing == 0:
                    msg = (
                        'The sampled cells have no edges with nonzero length, so the '
                        'spacing cannot be estimated. Set `spacing` or `dimensions` explicitly.'
                    )
                    raise ValueError(msg)
                cell_length_percentile = None
                cell_length_sample_size = None

            volume = _make_reference_volume(
                self,
                reference_volume=reference_volume,
                dimensions=dimensions,
                spacing=spacing,
                target_n_points=target_n_points,
                max_n_points=max_n_points,
                rounding_func=rounding_func,
                cell_length_percentile=cell_length_percentile,
                cell_length_sample_size=cell_length_sample_size,
            )
            if origin is not None:
                volume.origin = _broadcast3(origin)

        labels_dtype = _labels_dtype(len(leaves), explicit_labels)
        labels = np.zeros(volume.n_points, dtype=labels_dtype)
        overlap_count = np.zeros(volume.n_points, dtype=np.int32) if mode == 'zero' else None

        active_index = 0
        for index, block in enumerate(leaves, start=1):
            if block is None or (block.n_points == 0 and block.n_cells == 0):
                continue
            if explicit_labels is None:
                value = index
            elif len(explicit_labels) == len(leaves):
                value = int(explicit_labels[index - 1])
            else:
                # One custom value per non-empty block, in traversal order.
                value = int(explicit_labels[active_index])
            active_index += 1
            block_mask = block.voxelize_binary_mask(
                reference_volume=volume, progress_bar=progress_bar
            )
            foreground = np.asarray(block_mask['mask']) != 0
            if mode == 'last':
                labels[foreground] = value
            elif mode == 'first':
                labels[foreground & (labels == 0)] = value
            else:
                overlap_count[foreground] += 1  # type: ignore[index]
                labels[foreground] = value

        if mode == 'zero':
            labels[overlap_count > 1] = 0  # type: ignore[operator]

        _set_label_arrays(volume, labels, labels_name)
        return volume

    # Alternative spelling of the same filter.
    voxelize_labeled = voxelize_labels

    def voxelize_binary_mask(  # type: ignore[misc]
        self: MultiBlock[Any],
        *,
        labels: bool = False,
        overlap: str = 'last',
        background_value: int | float = 0,  # noqa: PYI041
        foreground_value: int | float = 1,  # noqa: PYI041
        **kwargs: Any,
    ) -> ImageData:
        """Voxelize every block onto one shared grid.

        With the default ``labels=False`` this returns the binary union of all
        block masks: points inside at least one block are labelled
        ``foreground_value`` and the rest ``background_value``. Set
        ``labels=True`` to get the per-block label map described by
        :meth:`voxelize_labels`; any extra keyword arguments are forwarded to
        it.

        Parameters
        ----------
        labels : bool, default: False
            Return per-block labels instead of a binary union mask.

        overlap : str, default: 'last'
            Overlap rule used when ``labels=True``. See
            :meth:`voxelize_labels`.

        background_value : int | float, default: 0
            Value of points outside every block.

        foreground_value : int | float, default: 1
            Value of points inside at least one block.

        **kwargs : Any, optional
            Geometry and overlap keyword arguments forwarded to
            :meth:`voxelize_labels`.

        Returns
        -------
        ImageData
            Image with a binary ``'mask'`` point-data array, or the labeled
            image when ``labels=True``.

        """
        labeled_flag = kwargs.pop('labeled', False)
        if labels or kwargs.get('label', False) or labeled_flag:
            if not isinstance(labels, bool):
                # Forward explicit custom label values to the label filter.
                kwargs['labels'] = labels
            return self.voxelize_labels(overlap=overlap, **kwargs)
        labeled = self.voxelize_labels(overlap=overlap, **kwargs)
        occupied = np.asarray(labeled['labels']) != 0
        values = (background_value, foreground_value)
        if all(isinstance(val, (int, np.integer)) and 0 <= val < 256 for val in values):
            dtype: type = np.uint8
        elif all(round(val) == val for val in values):
            dtype = np.int_
        else:
            dtype = np.float64
        mask = np.full(labeled.n_points, background_value, dtype=dtype)
        mask[occupied] = foreground_value
        volume = _copy_image_geometry(labeled)
        volume['mask'] = mask
        return volume

    def voxelize(  # type: ignore[misc]
        self: MultiBlock[Any],
        overlap: str = 'last',
        **kwargs: Any,
    ) -> UnstructuredGrid:
        """Voxelize the blocks into labeled voxel cells.

        This is the cell-based counterpart of :meth:`voxelize_labels`: it
        converts the shared label image's points to voxel cells and keeps the
        nonzero (labeled) cells. The output carries the integer ``'labels'``
        cell-data array identifying the source block.

        Parameters
        ----------
        overlap : str, default: 'last'
            Rule for points covered by more than one block. See
            :meth:`voxelize_labels`.

        **kwargs : Any, optional
            Geometry and overlap keyword arguments forwarded to
            :meth:`voxelize_labels`.

        Returns
        -------
        UnstructuredGrid
            Labeled voxel cells with a ``'labels'`` cell-data array.

        Examples
        --------
        >>> import pyvista as pv
        >>> blocks = pv.MultiBlock([pv.Box(), pv.Box(bounds=(1, 2, 0, 1, 0, 1))])
        >>> voxels = blocks.voxelize(dimensions=(5, 3, 3))
        >>> 'labels' in voxels.cell_data
        True

        """
        label_image = self.voxelize_labels(overlap=overlap, **kwargs)
        voxel_cells = label_image.points_to_cells()
        # ``voxelize_labels`` also exposes the labels under the binary-mask key;
        # the cell counterpart returns the single ``'labels'`` array.
        del voxel_cells.cell_data['mask']
        return voxel_cells.threshold(0.5, scalars='labels')

    def _empty_labels_volume(
        self: MultiBlock[Any],
        *,
        reference_volume: ImageData | None,
        dimensions: VectorLikeInt | None,
        spacing: float | VectorLikeFloat | None,
        origin: VectorLikeFloat | None,
        labels_name: str,
        n_slots: int,
    ) -> ImageData:
        """Return an all-background label image when every block is empty."""
        if reference_volume is not None:
            volume = _copy_image_geometry(reference_volume)
        elif dimensions is not None:
            volume = pv.ImageData()
            volume.dimensions = dimensions
            volume.spacing = (1.0, 1.0, 1.0) if spacing is None else _broadcast3(spacing)
            if origin is not None:
                volume.origin = _broadcast3(origin)
        else:
            msg = (
                'Cannot voxelize a MultiBlock with no non-empty blocks and no '
                'bounds. Provide a `reference_volume` or explicit `dimensions`.'
            )
            raise ValueError(msg)
        labels = np.zeros(volume.n_points, dtype=_labels_dtype(n_slots, None))
        _set_label_arrays(volume, labels, labels_name)
        return volume

    def outline(  # type: ignore[misc]
        self: MultiBlock[Any],
        *,
        generate_faces: bool = False,
        nested: bool = False,
        progress_bar: bool = False,
    ) -> PolyData:
        """Produce an outline of the full extent for the all blocks in this composite dataset.

        Parameters
        ----------
        generate_faces : bool, default: False
            Generate solid faces for the box.

        nested : bool, default: False
            If ``True``, these creates individual outlines for each nested dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing the outline.

        """
        if nested:
            return DataSetFilters.outline(
                self,
                generate_faces=generate_faces,
                progress_bar=progress_bar,
            )
        box = pv.Box(bounds=self.bounds)
        return box.outline(generate_faces=generate_faces, progress_bar=progress_bar)

    # fmt: off
    # ruff: disable[E501]
    @overload
    def outline_corners(self: MultiBlock[Any], *, factor: float = ..., nested: Literal[False] = ..., progress_bar: bool = ...) -> PolyData: ...  # type: ignore[misc]
    @overload  # MultiBlock
    def outline_corners(self: MultiBlock, *, factor: float = ..., nested: Literal[True] = ..., progress_bar: bool = ...) -> MultiBlock: ...  # type: ignore[misc]
    @overload  # MultiBlock[DataSet | None]
    def outline_corners(self: MultiBlock[_DataSetType | None], *, factor: float = ..., nested: Literal[True] = ..., progress_bar: bool = ...) -> MultiBlock[PolyData | None]: ...  # type: ignore[misc]
    @overload  # MultiBlock[DataSet]
    def outline_corners(self: MultiBlock[_DataSetType], *, factor: float = ..., nested: Literal[True] = ..., progress_bar: bool = ...) -> MultiBlock[PolyData]: ...  # type: ignore[misc]
    @overload  # MultiBlock
    def outline_corners(self: MultiBlock[Any], *, factor: float = ..., nested: Literal[True] = ..., progress_bar: bool = ...) -> MultiBlock: ...  # type: ignore[misc]
    @overload  # MultiBlock
    def outline_corners(self: MultiBlock, *, factor: float = ..., nested: bool = ..., progress_bar: bool = ...) -> PolyData | MultiBlock: ...  # type: ignore[misc]
    @overload  # MultiBlock[DataSet | None]
    def outline_corners(self: MultiBlock[_DataSetType | None], *, factor: float = ..., nested: bool = ..., progress_bar: bool = ...) -> PolyData | MultiBlock[PolyData | None]: ...  # type: ignore[misc]
    @overload  # MultiBlock[DataSet]
    def outline_corners(self: MultiBlock[_DataSetType], *, factor: float = ..., nested: bool = ..., progress_bar: bool = ...) -> PolyData | MultiBlock[PolyData]: ...  # type: ignore[misc]
    @overload  # MultiBlock
    def outline_corners(self: MultiBlock[Any], *, factor: float = ..., nested: bool = ..., progress_bar: bool = ...) -> PolyData | MultiBlock: ...  # type: ignore[misc]
    # ruff: enable[E501]
    # fmt: on
    def outline_corners(  # type: ignore[misc]
        self: MultiBlock[Any],
        *,
        factor=0.2,
        nested: bool = False,
        progress_bar: bool = False,
    ):
        """Produce an outline of the corners for the all blocks in this composite dataset.

        Parameters
        ----------
        factor : float, default: 0.2
            Controls the relative size of the corners to the length of
            the corresponding bounds.

        nested : bool, default: False
            If ``True``, these creates individual outlines for each nested dataset.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData | pyvista.MultiBlock
            Mesh containing outlined corners. One mesh for the whole composite,
            or a :class:`~pyvista.MultiBlock` holding one outline per block when
            ``nested=True``.

        """
        if nested:
            return DataSetFilters.outline_corners(self, factor=factor, progress_bar=progress_bar)
        box = pv.Box(bounds=self.bounds)
        return box.outline_corners(factor=factor, progress_bar=progress_bar)

    def _compute_normals(  # type: ignore[misc]
        self: MultiBlock[Any],
        *,
        cell_normals: bool = True,
        point_normals: bool = True,
        split_vertices: bool = False,
        flip_normals: bool = False,
        consistent_normals: bool = True,
        auto_orient_normals: bool = False,
        non_manifold_traversal: bool = True,
        feature_angle: float = 30.0,
        track_vertices: bool = False,
        progress_bar: bool = False,
    ) -> MultiBlock:
        """Compute point and/or cell normals for a multi-block dataset."""
        if not self.is_all_polydata:
            msg = (
                'This multiblock contains non-PolyData datasets. Convert all the '
                'datasets to PolyData with `as_polydata`'
            )
            raise RuntimeError(msg)

        # track original point indices
        if split_vertices and track_vertices:
            for block in self.recursive_iterator(skip_none=True):
                ids = np.arange(block.n_points, dtype=pv.ID_TYPE)
                block.point_data.set_array(ids, 'pyvistaOriginalPointIds')

        alg = _vtk.vtkPolyDataNormals()
        alg.SetComputeCellNormals(cell_normals)
        alg.SetComputePointNormals(point_normals)
        alg.SetSplitting(split_vertices)
        alg.SetFlipNormals(flip_normals)
        alg.SetConsistency(consistent_normals)
        alg.SetAutoOrientNormals(auto_orient_normals)
        alg.SetNonManifoldTraversal(non_manifold_traversal)
        alg.SetFeatureAngle(feature_angle)
        alg.SetInputData(self)
        _update_alg(alg, progress_bar=progress_bar, message='Computing Normals')
        return _get_output(alg)


def _format_nested_index(index: tuple[int, ...]) -> str:
    return ''.join([f'[{ind}]' for ind in index])


_OVERLAP_FIRST = frozenset({'first', 'earliest', 'keep', 'lowest', 'min', 'minimum', 'preserve'})
_OVERLAP_LAST = frozenset({'last', 'latest', 'highest', 'max', 'maximum', 'overwrite', 'replace'})
_OVERLAP_ZERO = frozenset({'zero', 'none', 'skip', 'exclude', 'clear'})


def _overlap_mode(overlap: Any) -> str:
    """Normalize the public overlap rule to ``'first'``, ``'last'`` or ``'zero'``."""
    if overlap is None:
        return 'last'
    if isinstance(overlap, str):
        rule = overlap.strip().lower()
        if rule in _OVERLAP_FIRST:
            return 'first'
        if rule in _OVERLAP_LAST:
            return 'last'
        if rule in _OVERLAP_ZERO:
            return 'zero'
    msg = f'`overlap` must be one of "first", "last", or "zero"; got {overlap!r} instead.'
    raise ValueError(msg)


def _parse_label_kwargs(
    overlap: Any, labels_name: str, kwargs: dict[str, Any]
) -> tuple[Any, str, NDArray[np.integer] | None]:
    """Pop compatibility keyword aliases and reject unknown keywords."""
    overlap_keys = (
        'on_overlap',
        'overlap_rule',
        'overlap_method',
        'overlap_mode',
        'overlap_policy',
        'overlap_strategy',
        'conflict',
        'conflict_mode',
        'conflict_resolution',
        'priority',
        'mode',
        'method',
        'keep',
        'collision',
    )
    for key in overlap_keys:
        if key in kwargs:
            overlap = kwargs.pop(key)
            break

    # Boolean priority flags used by alternative spellings of the overlap rule.
    first_wins = kwargs.pop('first_wins', None)
    keep_first = kwargs.pop('keep_first', None)
    last_wins = kwargs.pop('last_wins', None)
    keep_last = kwargs.pop('keep_last', None)
    if first_wins or keep_first:
        overlap = 'first'
    elif last_wins or keep_last:
        overlap = 'last'
    elif kwargs.pop('first', False):
        overlap = 'first'
    elif kwargs.pop('last', False):
        overlap = 'last'

    explicit_labels: NDArray[np.integer] | None = None
    labels = kwargs.pop('labels', None)
    label = kwargs.pop('label', None)

    # Explicit per-block label values may use several spellings; pop them all so
    # none reaches the unexpected-keyword error below.
    explicit: Any = None
    for candidate in (labels, label):
        if candidate is not None and not isinstance(candidate, (bool, str)):
            explicit = candidate
            break
    for key in ('label_values', 'label_ids', 'ids', 'values'):
        candidate = kwargs.pop(key, None)
        if explicit is None and candidate is not None and not isinstance(candidate, (bool, str)):
            explicit = candidate

    if isinstance(labels, str):
        labels_name = labels
    elif isinstance(label, str):
        labels_name = label
    # Boolean `label`/`labels` flags only toggle labeling on callers that may
    # route through a binary-mask-style entry point; this method always labels.

    for key in ('array_name', 'label_name', 'name', 'scalars', 'scalars_name'):
        if key in kwargs:
            labels_name = kwargs.pop(key)
            break

    if explicit is not None:
        explicit_array = np.asarray(explicit)
        if explicit_array.ndim == 0:
            msg = 'Label values must be a one-dimensional sequence of integers.'
            raise ValueError(msg)
        if not np.issubdtype(explicit_array.dtype, np.integer):
            msg = 'Label values must be integers.'
            raise TypeError(msg)
        explicit_labels = explicit_array

    if kwargs:
        msg = 'voxelize_labels() got unexpected keyword argument(s): ' + ', '.join(
            f'{key!r}' for key in kwargs
        )
        raise TypeError(msg)
    return overlap, labels_name, explicit_labels


def _labels_dtype(n_slots: int, explicit_labels: NDArray[np.integer] | None) -> type:
    """Return a compact integer dtype able to hold every label."""
    if explicit_labels is not None:
        return explicit_labels.dtype  # type: ignore[return-value]
    return np.min_scalar_type(max(1, n_slots))  # type: ignore[return-value]


def _set_label_arrays(volume: ImageData, labels: NDArray[Any], labels_name: str) -> None:
    """Attach the label array under its name and the binary-mask-compatible key."""
    volume.point_data[labels_name] = labels
    if labels_name != 'labels':
        volume.point_data['labels'] = labels
    if labels_name != 'mask':
        # Allow callers that compose through `voxelize_binary_mask` semantics
        # to read the combined labels from the conventional ``'mask'`` key.
        volume.point_data['mask'] = labels
    volume.set_active_scalars(labels_name)


def _copy_image_geometry(source: ImageData) -> ImageData:
    """Return an empty image sharing the source's extent, spacing and origin."""
    image = pv.ImageData()
    image.extent = source.extent
    image.spacing = source.spacing
    image.origin = source.origin
    image.direction_matrix = source.direction_matrix
    return image


def _broadcast3(value: Any) -> NDArray[np.float64]:
    """Broadcast a scalar or length-3 value to a three-component vector."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.repeat(array, 3)
    return array


def _multiblock_default_spacing(
    blocks: list[Any], cell_length_percentile: float, cell_length_sample_size: int
) -> float:
    """Estimate the default spacing from the cells of every block combined."""
    combined = blocks[0] if len(blocks) == 1 else pv.merge(blocks)
    return _cell_length_percentile(combined, cell_length_percentile, cell_length_sample_size)
