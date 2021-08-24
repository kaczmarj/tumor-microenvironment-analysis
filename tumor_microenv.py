"""Perform quantitative analysis of tumor microenvironment.

1. Read in JSON data. This gives us coordinates for each type of geometry object,
including cell polygons and heatmaps.
2. Store the minimum amount of data necessary for each patch
  - coordinates
  - patch type
  - biomarker status
3. For cell polygons, store
  - coordinates
  - cell type
4. Create a shapely geometric object (aka geom) for each label.
  - We should read all geoms as Polygons.
  - The coordinates of the heatmap can be passed to shapely's box function.
  - Assign patch type (tumor, non-tumor, blank) and biomarker status (n/a, pos, neg).
6. Create a union multipolygon for the tumor-positive patches.
5. Create a union multipolygon for the biomarker-positive patches.
6. Create a union multipolygon for the biomarker-negative patches.
7. Create a tumor microenvironment multipolygon... (dilate the union multipolygon).
8. For each tumor microenvironment, get all cell points contained inside.
  - 8a. For each point of each cell, find nearest biomarker-positive patch and nearest
      biomarker-negative patch.
  - 8b. Optionally, for each point of each cell, get lines connecting point and the
      nearest biomarker-positive and biomarker-negative patches.
"""

import csv
import enum
import json
from pathlib import Path
import typing as ty
import uuid

import numpy as np
from shapely.geometry import box as box_constructor
from shapely.geometry import JOIN_STYLE
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.ops import unary_union

PathType = ty.Union[str, bytes, Path]


class _PatchType(enum.IntEnum):
    BLANK = 0
    TUMOR = 1
    NONTUMOR = 2


class _BiomarkerStatus(enum.IntEnum):
    NA = 0  # tiles that do not contain tumor will be N/A for marker.
    POSITIVE = 1
    NEGATIVE = 2


class Patch(ty.NamedTuple):
    polygon: Polygon
    patch_type: _PatchType
    biomarker_status: _BiomarkerStatus


class _Cell(ty.NamedTuple):
    polygon: Polygon
    cell_type: str
    lattice_points: MultiPoint
    uuid: str


Patches = ty.List[Patch]
_Cells = ty.List[_Cell]


class _PosNegDistances(ty.NamedTuple):
    dpositive: float
    dnegative: float


class _PosNegLines(ty.NamedTuple):
    line_to_positive: LineString
    line_to_negative: LineString


class _PointOutputData(ty.NamedTuple):
    """Object representing one row in the output file."""

    point: str
    dist_to_marker_neg: float
    dist_to_marker_pos: float
    line_to_marker_neg: str
    line_to_marker_pos: str
    cell_uuid: str
    microenv_micrometer: int


def _patch_to_patch_type_and_biomarker_status(
    patch,
    background: int,
    marker_positive: int,
    marker_negative: int,
) -> ty.Tuple[_PatchType, _BiomarkerStatus]:
    """Return the patch type and biomarker status of the patch."""
    patch = np.asarray(patch)
    marker_pos_mask = patch == marker_positive
    marker_neg_mask = patch == marker_negative
    tumor_mask = np.logical_or(marker_pos_mask, marker_neg_mask)
    blank_mask = patch == background
    percent_tumor = tumor_mask.mean()
    percent_blank = blank_mask.mean()
    if percent_tumor > 0.5:
        if marker_pos_mask.mean() > marker_neg_mask.mean():
            return _PatchType.TUMOR, _BiomarkerStatus.POSITIVE
        else:
            return _PatchType.TUMOR, _BiomarkerStatus.NEGATIVE
    elif percent_blank > 0.5:
        return _PatchType.BLANK, _BiomarkerStatus.NA
    else:
        return _PatchType.NONTUMOR, _BiomarkerStatus.NA


def _path_to_polygon(path) -> Polygon:
    """Return a rectangular shapely Polygon from coordinates in a file name.

    Assumes the file is named `MINX_MINY_COLS_ROWS.EXT`.
    """
    path = Path(path)
    values = path.stem.split("_")
    # convert to int.
    minx, miny, cols, rows = map(int, values)
    maxx = minx + cols
    maxy = miny + rows
    return box_constructor(minx=minx, miny=miny, maxx=maxx, maxy=maxy)


def _npy_file_to_patch_object(
    path, background: int, marker_positive: int, marker_negative: int
) -> Patch:
    """Create a Patch object from a NPY file of segmentation results.

    Parameters
    ----------
    path : str, pathlib.Path
        Path to .npy file with segmentation results.
    background : int
        Value assigned to background pixels.
    marker_positive : int
        Value assigned to pixels positive for the biomarker.
    marker_negative : int
        Value assigned to pixels negative for the biomarker.

    Returns
    -------
    Patch object.
    """
    arr = np.load(path)
    polygon = _path_to_polygon(path)
    patch_type, biomarker_status = _patch_to_patch_type_and_biomarker_status(
        arr,
        background=background,
        marker_positive=marker_positive,
        marker_negative=marker_negative,
    )
    patch = Patch(
        polygon=polygon, patch_type=patch_type, biomarker_status=biomarker_status
    )
    return patch


def _get_tumor_microenvironment(
    tumor_geom: MultiPolygon, distance: int
) -> MultiPolygon:
    """Return a dilated MultiPolygon of tumors, representing the microenvironment at a
    given distance.
    """
    # mitre will join dilated squares as squares.
    return tumor_geom.buffer(distance=distance, join_style=JOIN_STYLE.mitre)


def _get_distances_for_point(
    point: Point, positive_patches, negative_patches
) -> _PosNegDistances:
    """Get the distances from the point to the nearest positive and negative patches."""
    dpos = point.distance(positive_patches)
    dneg = point.distance(negative_patches)
    return _PosNegDistances(dpositive=dpos, dnegative=dneg)


def _get_nearest_points_for_point(
    point: Point, positive_patches, negative_patches
) -> _PosNegLines:
    """Get the lines joining the point to the nearest positive patch and the nearest
    negative patch.
    """
    line_to_pos = nearest_points(point, positive_patches)
    line_to_neg = nearest_points(point, negative_patches)
    line_to_pos = LineString(line_to_pos)
    line_to_neg = LineString(line_to_neg)
    return _PosNegLines(line_to_positive=line_to_pos, line_to_negative=line_to_neg)


def load_patches_and_cells(
    patch_paths: ty.List[PathType],
    cells_json: PathType,
    background: int,
    marker_positive: int,
    marker_negative: int,
) -> ty.Tuple[Patches, _Cells]:
    """Create list of Patches and Cells from patch paths and a JSON of cell data.

    Returns
    -------
    Tuple of (patches, cells).
    """
    patches: Patches = []
    for patch_path in patch_paths:
        patches.append(
            _npy_file_to_patch_object(
                path=patch_path,
                background=background,
                marker_positive=marker_positive,
                marker_negative=marker_negative,
            )
        )

    with open(cells_json) as f:
        cells_data: ty.List[ty.Dict[str, ty.Any]] = json.load(f)

    cells: _Cells = []
    for cell_data in cells_data:
        cells.append(
            _Cell(
                polygon=Polygon(cell_data["coordinates"]),
                cell_type=cell_data["type"],
                lattice_points=MultiPoint(cell_data["lattice_points"]),
                uuid=uuid.uuid4().hex,
            )
        )

    return patches, cells


def _distances_for_cell_in_microenv(
    cell: _Cell,
    marker_positive_geom,
    marker_negative_geom,
    microenv_micrometer: int,
) -> ty.Generator[_PointOutputData, None, None]:
    """Yield distance information for one cell."""
    for cell_point in cell.lattice_points.geoms:
        distances = _get_distances_for_point(
            cell_point,
            positive_patches=marker_positive_geom,
            negative_patches=marker_negative_geom,
        )
        lines_from_point_to_patches = _get_nearest_points_for_point(
            cell_point,
            positive_patches=marker_positive_geom,
            negative_patches=marker_negative_geom,
        )
        yield _PointOutputData(
            point=cell_point.wkt,
            dist_to_marker_neg=distances.dnegative,
            dist_to_marker_pos=distances.dpositive,
            line_to_marker_neg=lines_from_point_to_patches.line_to_negative.wkt,
            line_to_marker_pos=lines_from_point_to_patches.line_to_positive.wkt,
            cell_uuid=cell.uuid,
            microenv_micrometer=microenv_micrometer,
        )


def run_spatial_analysis(
    patches: Patches,
    cells: _Cells,
    microenv_distances: ty.Sequence[int],
    output_path: PathType = "output.csv",
):
    """Run spatial analysis workflow."""
    # This is a multipolygon that represents the entire tumor in our region of interest.
    tumor_geom: MultiPolygon = unary_union(
        [p.polygon for p in patches if p.patch_type == _PatchType.TUMOR]
    )
    marker_positive_patches = (
        p.polygon for p in patches if p.biomarker_status == _BiomarkerStatus.POSITIVE
    )
    marker_negative_patches = (
        p.polygon for p in patches if p.biomarker_status == _BiomarkerStatus.NEGATIVE
    )
    marker_positive_geom = unary_union(list(marker_positive_patches))
    marker_negative_geom = unary_union(list(marker_negative_patches))
    del marker_positive_patches, marker_negative_patches

    d = 25  # TODO: we need to convert micrometers to pixels.

    with open(output_path, "w", newline="") as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=_PointOutputData._fields)
        dict_writer.writeheader()

        for distance in microenv_distances:
            tumor_microenv = _get_tumor_microenvironment(
                tumor_geom=tumor_geom, distance=distance
            )

            # TODO: this is NOT the same as the method we discussed with Joel and
            # Mahmudul. We discussed getting all of the POINTS inside the
            # microenvironment. But here, we take all of the CELLS in the
            # microenvironment. It's much easier to implement this, so let's roll with
            # it.
            cells_in_microenv = (
                cell for cell in cells if tumor_microenv.contains(cell.polygon)
            )
            for cell in cells_in_microenv:
                row_generator = _distances_for_cell_in_microenv(
                    cell=cell,
                    marker_positive_geom=marker_positive_geom,
                    marker_negative_geom=marker_negative_geom,
                    microenv_micrometer=d,
                )
                for row in row_generator:
                    dict_writer.writerow(row._asdict())
