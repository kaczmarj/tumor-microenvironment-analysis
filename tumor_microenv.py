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

import enum
import itertools
import json
import typing as ty

from shapely.geometry import JOIN_STYLE
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.ops import unary_union


class _PatchType(enum.IntEnum):
    BLANK = 0
    TUMOR = 1
    NONTUMOR = 2


class _BiomarkerStatus(enum.IntEnum):
    NA = 0  # tiles that do not contain tumor will be N/A for marker.
    POSITIVE = 1
    NEGATIVE = 2


class _Patch(ty.NamedTuple):
    polygon: Polygon
    patch_type: _PatchType
    biomarker_status: _BiomarkerStatus


class _Cell(ty.NamedTuple):
    polygon: Polygon
    cell_type: str
    lattice_points: MultiPoint


_Patches = ty.List[_Patch]
_Cells = ty.List[_Cell]


class _PosNegDistances(ty.NamedTuple):
    dpositive: float
    dnegative: float


class _PosNegLines(ty.NamedTuple):
    line_to_positive: ty.Tuple[Point, Point]
    line_to_negative: ty.Tuple[Point, Point]


def _standardize_patch_type(s: str) -> _PatchType:
    d = {
        "tumor": _PatchType.TUMOR,
        "non-tumor": _PatchType.NONTUMOR,
        "blank": _PatchType.BLANK,
    }
    try:
        return d[s.lower()]
    except KeyError:
        raise KeyError(f"unknown patch type: {s}")


def _standardize_biomarker_status(s: str) -> _BiomarkerStatus:
    d = {
        "na": _BiomarkerStatus.NA,
        "positive": _BiomarkerStatus.POSITIVE,
        "negative": _BiomarkerStatus.NEGATIVE,
    }
    try:
        return d[s.lower()]
    except KeyError:
        raise KeyError(f"unknown biomarker status: {s}")


def _load_patches_and_cells(path: str) -> ty.Tuple[_Patches, _Cells]:
    """Load JSON data of labels into lists of patches and cells.

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    Tuple of (patches, cells).
    """
    with open(path) as f:
        labels: ty.List[ty.Dict[str, ty.Any]] = json.load(f)

    patches: _Patches = []
    cells: _Cells = []
    for label in labels:
        # This will need to be modified.
        if label["name"] == "heatmap":
            this_geom = Polygon(label["coordinates"])
            this_patch_type = _standardize_patch_type(label["type"])
            this_biomarker_status = _standardize_biomarker_status(label["bm"])
            this_patch = _Patch(
                polygon=this_geom,
                patch_type=this_patch_type,
                biomarker_status=this_biomarker_status,
            )
            patches.append(this_patch)
            del this_geom, this_patch_type, this_biomarker_status, this_patch
        elif label["name"] == "cell":
            this_geom = Polygon(label["coordinates"])
            this_cell_type = label["type"]
            this_cell_lattice = _geom_to_lattice_points(this_geom)
            this_cell = _Cell(
                polygon=this_geom,
                cell_type=this_cell_type,
                lattice_points=this_cell_lattice,
            )
            cells.append(this_cell)
            del this_geom, this_cell_type, this_cell_lattice, this_cell
    return patches, cells


def _get_tumor_microenvironment(
    tumor_geom: MultiPolygon, distance: int
) -> MultiPolygon:
    """Return a dilated MultiPolygon of tumors, representing the microenvironment at a
    given distance.
    """
    # mitre will join dilated squares as squares.
    return tumor_geom.buffer(distance=distance, join_style=JOIN_STYLE.mitre)


def _geom_to_lattice_points(geom) -> MultiPoint:
    """Return lattice points of a shapely object.

    Parameters
    ---------
    geom : shapely geometric object

    Returns
    -------
    shapely.MultiPoint instance containing the points that make up the input object.
    """
    # convert values to int
    xmin, ymin, xmax, ymax = map(round, geom.bounds)
    coords = itertools.product(range(xmin, xmax + 1), range(ymin, ymax + 1))
    points = (Point(*p) for p in coords)
    points = MultiPoint(points).intersection(geom)
    return points


def _get_distances_for_point(
    point: Point, positive_patches: _Patches, negative_patches: _Patches
) -> _PosNegDistances:
    """Get the distances from the point to the nearest positive and negative patches."""
    dpos = point.distance(positive_patches)
    dneg = point.distance(negative_patches)
    return _PosNegDistances(dpositive=dpos, dnegative=dneg)


def _get_nearest_points_for_point(
    point: Point, positive_patches: _Patches, negative_patches: _Patches
) -> _PosNegLines:
    """Get the lines joining the point to the nearest positive patch and the nearest
    negative patch.
    """
    line_to_pos = nearest_points(point, positive_patches)
    line_to_neg = nearest_points(point, negative_patches)
    return _PosNegLines(line_to_positive=line_to_pos, line_to_negative=line_to_neg)


def run_spatial_analysis(path: str):
    patches, cells = _load_patches_and_cells(path)
    tumor_geom: MultiPolygon = unary_union(
        [p.polygon for p in patches if p.patch_type == _PatchType.TUMOR]
    )

    microenv_distance = 25  # TODO: we need to convert micrometers to pixels.
    tumor_microenv = _get_tumor_microenvironment(
        tumor_geom=tumor_geom, distance=microenv_distance
    )
    # TODO: this is NOT the same as the method we discussed with Joel and Mahmudul.
    # We discussed getting all of the POINTS inside the microenvironment. But here,
    # we take all of the CELLS in the microenvironment. It's much easier to implement
    # this, so let's roll with it.
    cells_in_microenv = (
        cell for cell in cells if tumor_microenv.contains(cell.polygon)
    )

    # TODO: CONTINUE FROM HERE...
