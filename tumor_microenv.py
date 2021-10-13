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
import itertools
import json
from pathlib import Path
import random
import typing as ty
import uuid

import numpy as np
from shapely.geometry import base as _base_geometry
from shapely.geometry import box as box_constructor
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.ops import unary_union
from tqdm import tqdm

PathType = ty.Union[str, Path]


class PatchType(enum.IntEnum):
    BLANK = 0
    TUMOR = 1
    NONTUMOR = 2


class BiomarkerStatus(enum.IntEnum):
    NA = 0  # tiles that do not contain tumor will be N/A for marker.
    POSITIVE = 1
    NEGATIVE = 2


class Patch(ty.NamedTuple):
    polygon: _base_geometry.BaseGeometry
    patch_type: PatchType
    biomarker_status: BiomarkerStatus


class Cell(ty.NamedTuple):
    polygon: _base_geometry.BaseGeometry
    cell_type: str
    uuid: str

    @property
    def lattice_points(self) -> MultiPoint:
        """Lattice points of the polygon (all integer points that make up polygon)."""
        if isinstance(self.polygon, Point):
            return MultiPoint([self.polygon])
        # convert values to int
        xmin, ymin, xmax, ymax = map(round, self.polygon.bounds)
        coords = itertools.product(range(xmin, xmax + 1), range(ymin, ymax + 1))
        points = (Point(*p) for p in coords)
        points = MultiPoint(list(points)).intersection(self.polygon)
        return points


Patches = ty.List[Patch]
Cells = ty.List[Cell]


class PosNegDistances(ty.NamedTuple):
    # In some cases, a biomarker-negative or -positive region might be so far away
    # that we exclude that distance. In those cases, the distance would be None.
    dpositive: ty.Optional[float] = None
    dnegative: ty.Optional[float] = None


class PosNegLines(ty.NamedTuple):
    # See comment in PosNegDistances regarding None values.
    line_to_positive: ty.Optional[LineString] = None
    line_to_negative: ty.Optional[LineString] = None


class PointOutputData(ty.NamedTuple):
    """Object representing one row in the output file."""

    point: str
    cell_type: str
    cell_uuid: str
    microenv_micrometer: int
    # See comment in PosNegDistances regarding None values.
    dist_to_marker_neg: ty.Optional[float] = None
    dist_to_marker_pos: ty.Optional[float] = None
    line_to_marker_neg: ty.Optional[str] = None
    line_to_marker_pos: ty.Optional[str] = None


def _get_distances_for_point(
    point: Point,
    positive_geom: ty.Union[_base_geometry.BaseGeometry, None],
    negative_geom: ty.Union[_base_geometry.BaseGeometry, None],
) -> PosNegDistances:
    """Get the distances from the point to the nearest positive and negative patches."""
    dpos = point.distance(positive_geom) if positive_geom is not None else None
    dneg = point.distance(negative_geom) if negative_geom is not None else None
    return PosNegDistances(dpositive=dpos, dnegative=dneg)


def _get_nearest_points_for_point(
    point: Point,
    positive_geom: ty.Union[_base_geometry.BaseGeometry, None],
    negative_geom: ty.Union[_base_geometry.BaseGeometry, None],
) -> PosNegLines:
    """Get the lines joining the point to the nearest positive patch and the nearest
    negative patch.
    """
    if positive_geom is not None:
        line_to_pos = nearest_points(point, positive_geom)
        line_to_pos = LineString(line_to_pos)
    else:
        line_to_pos = None
    if negative_geom is not None:
        line_to_neg = nearest_points(point, negative_geom)
        line_to_neg = LineString(line_to_neg)
    else:
        line_to_neg = None
    return PosNegLines(line_to_positive=line_to_pos, line_to_negative=line_to_neg)


def _get_exterior_of_geom(
    geom: ty.Union[Polygon, MultiPolygon, GeometryCollection],
) -> MultiLineString:
    if hasattr(geom, "geoms"):
        return MultiLineString([g.exterior for g in geom.geoms])
    else:
        return MultiLineString([geom.exterior])


def _exterior_to_multilinestring(
    exterior: MultiLineString,
) -> MultiLineString:
    lines: ty.List[ty.Tuple[ty.Tuple[float, float], ty.Tuple[float, float]]] = []
    for t in exterior:
        lines.extend(zip(t.coords[:-1], t.coords[1:]))
    return MultiLineString(lines)


def _get_exterior_contained_in_larger_geom(
    multigeom: _base_geometry.BaseMultipartGeometry,
    larger_geom_exterior: MultiLineString,
) -> MultiLineString:
    """Get the exterior lines of `multigeom` that are contained in the exterior lines
    of `larger_geom_exterior`.
    """
    exterior = _get_exterior_of_geom(multigeom)
    lines = _exterior_to_multilinestring(exterior)
    # Buffer just in case, so we definitely contain the line (?)
    larger_geom_exterior = larger_geom_exterior.buffer(1)
    return MultiLineString([g for g in lines if larger_geom_exterior.contains(g)])


def _distances_for_cell_in_microenv(
    cell: Cell,
    marker_positive_geom: _base_geometry.BaseGeometry,
    marker_negative_geom: _base_geometry.BaseGeometry,
    microenv_micrometer: int,
) -> ty.Generator[PointOutputData, None, None]:
    """Yield distance information for one cell."""
    cell_point: Point
    for cell_point in cell.lattice_points.geoms:
        distances = _get_distances_for_point(
            cell_point,
            positive_geom=marker_positive_geom,
            negative_geom=marker_negative_geom,
        )
        try:
            lines_from_point_to_patches = _get_nearest_points_for_point(
                cell_point,
                positive_geom=marker_positive_geom,
                negative_geom=marker_negative_geom,
            )
        # Sometimes this can error... but why?
        except ValueError:
            continue

        if lines_from_point_to_patches.line_to_negative is None:
            line_to_marker_neg = None
        else:
            line_to_marker_neg = lines_from_point_to_patches.line_to_negative.wkt
        if lines_from_point_to_patches.line_to_positive is None:
            line_to_marker_pos = None
        else:
            line_to_marker_pos = lines_from_point_to_patches.line_to_positive.wkt

        yield PointOutputData(
            point=cell_point.wkt,
            dist_to_marker_neg=distances.dnegative,
            dist_to_marker_pos=distances.dpositive,
            line_to_marker_neg=line_to_marker_neg,
            line_to_marker_pos=line_to_marker_pos,
            cell_type=cell.cell_type,
            cell_uuid=cell.uuid,
            microenv_micrometer=microenv_micrometer,
        )


def get_exteriors(
    tumor: _base_geometry.BaseMultipartGeometry,
    biomarker_positive: ty.Union[_base_geometry.BaseGeometry, None],
    biomarker_negative: ty.Union[_base_geometry.BaseGeometry, None],
) -> ty.Dict[str, ty.Union[MultiLineString, None]]:
    """Get exteriors of tumor, biomarker-positive, and biomarker-negative polygons."""
    tumor_exterior = _get_exterior_of_geom(tumor)
    if biomarker_positive is not None:
        marker_positive_exterior = _get_exterior_contained_in_larger_geom(
            biomarker_positive, tumor_exterior
        )
    else:
        marker_positive_exterior = None
    if biomarker_negative is not None:
        marker_negative_exterior = _get_exterior_contained_in_larger_geom(
            biomarker_negative, tumor_exterior
        )
    else:
        marker_negative_exterior = None
    return dict(
        tumor=tumor_exterior,
        marker_positive=marker_positive_exterior,
        marker_negative=marker_negative_exterior,
    )



def get_tumor_instance_and_influence_area(patches, multi_tumor_polygon, marker_positive_geom, marker_negative_geom, output_path, microenv_micrometer, patch_size = 73):
    with open(output_path, "w", newline="") as output_csv:
        patch_dict_writer = csv.DictWriter(output_csv, fieldnames=PointOutputData._fields)
        patch_dict_writer.writeheader()
        for patch in tqdm(patches):
            x, y = patch.polygon.exterior.coords.xy
            x, y = int(x[3]), int(y[3])
            mid_x, mid_y = int(x + patch_size //2), int(y + patch_size//2)
            point = Point(mid_x, mid_y)
            if patch.patch_type == PatchType.TUMOR:
                continue
            else:
                if not multi_tumor_polygon.contains(point):
                    distances = _get_distances_for_point(point, positive_geom=marker_positive_geom, negative_geom=marker_negative_geom)
                    try:
                        lines_from_point_to_patches = _get_nearest_points_for_point(point, positive_geom=marker_positive_geom, negative_geom=marker_negative_geom)
                    except ValueError:
                        continue

                    if lines_from_point_to_patches.line_to_negative is None:
                        line_to_marker_neg = None
                    else:
                        line_to_marker_neg = lines_from_point_to_patches.line_to_negative.wkt
                    
                    if lines_from_point_to_patches.line_to_positive is None:
                        line_to_marker_pos = None
                    else:
                        line_to_marker_pos = lines_from_point_to_patches.line_to_positive.wkt

                    patch_row = PointOutputData(point=point.wkt, dist_to_marker_neg=distances.dnegative, dist_to_marker_pos=distances.dpositive, line_to_marker_neg=line_to_marker_neg,
                        line_to_marker_pos=line_to_marker_pos, cell_type= "patch", cell_uuid="patch_id", microenv_micrometer=microenv_micrometer,)
                    patch_dict_writer.writerow(patch_row._asdict())
            

def run_spatial_analysis(
    patches: Patches,
    cells: Cells,
    microenv_distances: ty.Sequence[int],
    mpp: float,
    output_path: PathType = "output.csv",
    output_patch_path: PathType = "output_patches.csv",
    progress_bar: bool = True,
    patch_size: int = 73
):
    """Run spatial analysis workflow.

    Results are stored in a CSV file.

    Parameters
    ----------
    patches : list of Patch instances
    cells : list of Cell instances
    microenv_distances : sequence of int
        Distances (in micrometers) to consider for tumor microenvironment.
    output_path : PathType
        Path to output CSV.
    progress_bar : bool
        Whether to show the progress bar.
    """
    # This is a multipolygon that represents the entire tumor in our region of interest.
    tumor_patches = [p.polygon for p in patches if p.patch_type == PatchType.TUMOR]
    if not tumor_patches:
        print("no tumor patches found...")
        return
    tumor_geom = unary_union(tumor_patches)
    marker_positive_patches = [
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.POSITIVE
    ]
    marker_negative_patches = [
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.NEGATIVE
    ]
    if marker_positive_patches:
        marker_positive_geom = unary_union(marker_positive_patches)
    else:
        marker_positive_geom = None
    if marker_negative_patches:
        marker_negative_geom = unary_union(marker_negative_patches)
    else:
        marker_negative_geom = None

    exteriors = get_exteriors(
        tumor=tumor_geom,
        biomarker_positive=marker_positive_geom,
        biomarker_negative=marker_negative_geom,
    )
    tumor_exterior: MultiLineString = exteriors["tumor"]
    marker_positive_geom = exteriors["marker_positive"]
    marker_negative_geom = exteriors["marker_negative"]
    del marker_positive_patches, marker_negative_patches

    tumor_polygon = []
    for line in tumor_exterior:
        multi_tumor_polygon = MultiPolygon(tumor_polygon)
        new_poly = Polygon(line.coords)
        if multi_tumor_polygon.contains(new_poly):
            print("Tumor Polygon found inside another one")
        else:
            tumor_polygon.append(new_poly)
    multi_tumor_polygon = MultiPolygon(tumor_polygon)    

    with open(output_path, "w", newline="") as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=PointOutputData._fields)
        dict_writer.writeheader()

        for distance_um in microenv_distances:
            # get_tumor_instance_and_influence_area(patches, multi_tumor_polygon, marker_positive_geom, marker_negative_geom, output_patch_path, distance_um)
            distance_px = round(distance_um / mpp)
            print(f"Working on distance = {distance_um} um ({distance_px} px)")
            tumor_microenv = tumor_exterior.buffer(distance=distance_px)
            # TODO: this is NOT the same as the method we discussed with Joel and
            # Mahmudul. We discussed getting all of the POINTS inside the
            # microenvironment. But here, we take all of the CELLS in the
            # microenvironment. It's much easier to implement this, so let's roll with
            # it.
            # TODO: another way of finding cells that are near tumor is to query whether
            # the distance of each cell is less than our tumor microenvironment. This
            # would probably be better than buffering, because the buffer function
            # seems to introduce some artifacts.
            # print("Filtering cells in tumor microenvironment...")
            cells_in_microenv = [
                cell
                for cell in cells
                if tumor_microenv.contains(cell.polygon)
                and not multi_tumor_polygon.contains(cell.polygon)
            ]
            # print("Calculating distances for each cell...")
            for cell in tqdm(cells_in_microenv, disable=not progress_bar):
                row_generator = _distances_for_cell_in_microenv(
                    cell=cell,
                    marker_positive_geom=marker_positive_geom,
                    marker_negative_geom=marker_negative_geom,
                    microenv_micrometer=distance_um,
                )
                for row in row_generator:
                    dict_writer.writerow(row._asdict())



class BaseLoader:
    """BaseLoader object.

    The purpose of this object is to provide a uniform method of creating a list of
    Patch objects and a list of Cell objects. The __call__() method of this object
    must return a tuple of (Patches, Cells).
    """

    def __call__(self) -> ty.Tuple[Patches, Cells]:
        out = self.load()
        if len(out) != 2:
            raise ValueError("expected self.load() to return three objects")
        patches, cells = out
        if not all(isinstance(p, Patch) for p in patches):
            raise ValueError("first return val of self.load() must be list of Patches")
        if not all(isinstance(c, Cell) for c in cells):
            raise ValueError("second return val of self.load() must be list of Cells")
        return patches, cells

    def load(self) -> ty.Tuple[Patches, Cells]:
        raise NotImplementedError()


class LoaderV1(BaseLoader):
    """First iteration of data scheme.

    We have numpy files (.npy) with segmentation results and a JSON file with data about
    cells.

    Classes
    -------
    1 : k17 positive
    2 : cd8
    3 : cd16
    4 : cd4
    5 : cd3
    6 : cd163
    7 : k17 negative

    Assumptions
    -----------
    - .npy files are named `MINX_MINY_COLS_ROWS.EXT`.
    - JSON file with cell data has a list of objects. Each object must have the keys
        - coordinates
        - type
        - lattice_points
    """

    def __init__(
        self,
        patch_paths: ty.Sequence[PathType],
        cells_json: ty.Sequence[PathType],
        background: int,
        marker_positive: int,
        marker_negative: int,
        tumor_threshold: float = 0.05,
        marker_pos_threshold: float = 0.40,
    ):
        self.patch_paths = patch_paths
        self.cells_json = cells_json
        self.background = background
        self.marker_positive = marker_positive
        self.marker_negative = marker_negative
        self.tumor_threshold = tumor_threshold
        self.marker_pos_threshold = marker_pos_threshold

    @staticmethod
    def _path_to_polygon(path) -> Polygon:
        """Return a rectangular shapely Polygon from coordinates in a file name."""
        path = Path(path)
        values = path.stem.split("_")
        # convert to int.
        minx, miny, cols, rows = map(int, values)
        maxx = minx + cols
        maxy = miny + rows
        return box_constructor(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def _patch_to_patch_type_and_biomarker_status(
        self,
        patch,
    ) -> ty.Tuple[PatchType, BiomarkerStatus]:
        """Return the patch type and biomarker status of the patch."""
        patch = np.asarray(patch)
        marker_pos_mask = patch == self.marker_positive
        marker_neg_mask = patch == self.marker_negative
        tumor_mask = np.logical_or(marker_pos_mask, marker_neg_mask)
        percent_tumor = tumor_mask.mean()
        nonbackground_mask = patch != self.background
        percent_nonbackground = nonbackground_mask.mean()
        if percent_tumor >= self.tumor_threshold:
            n_tumor_points = tumor_mask.sum()
            if marker_pos_mask.sum() / n_tumor_points >= self.marker_pos_threshold:
                return PatchType.TUMOR, BiomarkerStatus.POSITIVE
            else:
                return PatchType.TUMOR, BiomarkerStatus.NEGATIVE
        elif percent_nonbackground > 0.01:
            return PatchType.NONTUMOR, BiomarkerStatus.NA
        else:
            return PatchType.BLANK, BiomarkerStatus.NA

    def _npy_file_to_patch_object(self, path: PathType) -> Patch:
        """Create a Patch object from a NPY file of segmentation results."""
        arr = np.load(path)
        polygon = self._path_to_polygon(path)
        patch_type, biomarker_status = self._patch_to_patch_type_and_biomarker_status(
            arr,
        )
        patch = Patch(
            polygon=polygon, patch_type=patch_type, biomarker_status=biomarker_status
        )
        return patch

    def _load_cells(self, path: PathType) -> ty.Generator[Cell, None, None]:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                coordinates = d["coordinates"]
                coordinates = [(int(x), int(y)) for x, y in d["coordinates"]]
                # TODO: we should support more than just polygons...
                if len(coordinates) < 3:
                    continue
                polygon = Polygon(coordinates)
                if not polygon.is_valid:
                    continue
                cell = Cell(
                    polygon=polygon,
                    cell_type=d["stain_class"],
                    uuid=uuid.uuid4().hex,
                )
                yield cell

    def load(self) -> ty.Tuple[Patches, Cells]:
        """Load data into a list of Patch objects and a list of Cell objects."""
        patches: Patches = []
        for patch_path in self.patch_paths:
            patches.append(self._npy_file_to_patch_object(path=patch_path))

        cells: Cells = []
        for cells_path in self.cells_json:
            for cell in self._load_cells(cells_path):
                cells.append(cell)

        return patches, cells


def read_point_csv(path: PathType) -> ty.List[PointOutputData]:
    """Read CSV of point informaation as a list of PointOutputData namedtuples."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [PointOutputData._make(r) for r in reader]


class CentralPatchFileNotFound(FileNotFoundError):
    """At least one central patch file is not found."""


def get_npy_and_json_files_for_roi(
    xmin: int,
    ymin: int,
    patch_size: int,
    analysis_size: int,
    tumor_microenv: int,
    data_root: PathType,
) -> ty.Tuple[ty.List[Path], ty.List[Path]]:
    """
    Get the paths to cells and the paths to npy patches that are within `tumor_microenv`
    pixels to the borders.

    Paramters
    ---------
    xmin, ymin : int
        Coordinates of the upper-left corner of the patch.
    patch_size : int
        Size of each patch in pixels. We assume patches are square.
    analysis_size : int
        Size of region (pixels) in which we consider cells. This is square.
    tumor_microenv : int
        Distance of tumor microenvironment in pixels.
    data_root : PathType
        Directory containing npy and json files.

    Returns
    -------
    Tuple of patch paths and cell (json) paths that are relevant for this ROI.
    """
    import math

    actual_analysis_size = patch_size * math.ceil(analysis_size / patch_size)
    actual_tumor_microenv = patch_size * math.ceil(tumor_microenv / patch_size)
    del analysis_size, tumor_microenv
    patches_right = patches_down = math.ceil(actual_analysis_size / patch_size)

    # These are the x and y coordinates of the upper-left corner of each patch in the
    # analysis region (ie the area from which we take cells).
    xs = [xmin + patch_size * i for i in range(patches_right)]
    ys = [ymin + patch_size * i for i in range(patches_down)]

    n_patches_for_tumor_env = math.ceil(actual_tumor_microenv / patch_size)

    left_x = [xs[0] + patch_size * i for i in range(-n_patches_for_tumor_env, 0)]
    left_y = [
        ys[0] + patch_size * i
        for i in range(-n_patches_for_tumor_env, patches_down + n_patches_for_tumor_env)
    ]

    right_x = [xs[-1] + patch_size * i for i in range(n_patches_for_tumor_env)]
    right_y = [
        ys[0] + patch_size * i
        for i in range(-n_patches_for_tumor_env, patches_down + n_patches_for_tumor_env)
    ]

    top_x = [
        xs[0] + patch_size * i
        for i in range(
            -n_patches_for_tumor_env, patches_right + n_patches_for_tumor_env
        )
    ]
    top_y = [ys[0] + patch_size * i for i in range(-n_patches_for_tumor_env, 0)]

    bottom_x = [
        xs[0] + patch_size * i
        for i in range(
            -n_patches_for_tumor_env, patches_right + n_patches_for_tumor_env
        )
    ]
    bottom_y = [ys[-1] + patch_size * i for i in range(n_patches_for_tumor_env)]

    assert top_x == bottom_x and left_y == right_y

    # Pad on both sides
    total_size = actual_analysis_size + 2 * actual_tumor_microenv
    # Check that our border coordinates make sense...
    # X coordinates
    assert total_size == top_x[-1] + patch_size - top_x[0], "top_x wrong"
    assert (
        actual_tumor_microenv == right_x[-1] + patch_size - right_x[0]
    ), "right_x wrong"
    assert total_size == bottom_x[-1] + patch_size - bottom_x[0], "bottom_x wrong"
    assert actual_tumor_microenv == left_x[-1] + patch_size - left_x[0], "left_x wrong"
    # Y coordinates
    assert actual_tumor_microenv == top_y[-1] + patch_size - top_y[0], "top_y wrong"
    assert total_size == right_y[-1] + patch_size - right_y[0], "right_y wrong"
    assert (
        actual_tumor_microenv == bottom_y[-1] + patch_size - bottom_y[0]
    ), "bottom_y wrong"
    assert total_size == left_y[-1] + patch_size - left_y[0], "left_y wrong"

    def coords_to_paths(
        xs,
        ys,
        patch_size: int,
        extension: str,
        parent: PathType = None,
    ) -> ty.List[Path]:
        """Convert coordinates to a path."""
        x_y_coords = itertools.product(xs, ys)
        paths = [
            Path(f"{x}_{y}_{patch_size}_{patch_size}.{extension}")
            for x, y in x_y_coords
        ]
        if parent is not None:
            parent = Path(parent)
            paths = [parent / p for p in paths]
        return paths

    # Patches that overlap with cells we consider.
    patches_in_roi = coords_to_paths(
        xs, ys, patch_size=patch_size, extension="npy", parent=data_root
    )
    if not all(p.exists() for p in patches_in_roi):
        raise CentralPatchFileNotFound("some central patches do not exist")

    top_patches = coords_to_paths(
        top_x, top_y, patch_size=patch_size, extension="npy", parent=data_root
    )
    right_patches = coords_to_paths(
        right_x, right_y, patch_size=patch_size, extension="npy", parent=data_root
    )
    bottom_patches = coords_to_paths(
        bottom_x, bottom_y, patch_size=patch_size, extension="npy", parent=data_root
    )
    left_patches = coords_to_paths(
        left_x, left_y, patch_size=patch_size, extension="npy", parent=data_root
    )

    # all_top_patches_exist = all(p.exists() for p in top_patches)
    # all_right_patches_exist = all(p.exists() for p in right_patches)
    # all_bottom_patches_exist = all(p.exists() for p in bottom_patches)
    # all_left_patches_exist = all(p.exists() for p in left_patches)

    # Some scenarios should never happen.
    # TODO: we need to fix this... at upper-left corner, for example, not all of the
    # right-most patches will exist. Specifically the top patches at the right won't
    # exist.
    # if not all_left_patches_exist and not all_right_patches_exist:
    #     raise FileNotFoundError("some left and right patches do not exist")
    # if not all_top_patches_exist and not all_bottom_patches_exist:
    #     raise FileNotFoundError("some top and bottom patches do not exist")

    # At corners, some patches will not exist.
    # if not all_top_patches_exist and not all_left_patches_exist:
    #     print("upper-left corner")
    # elif not all_top_patches_exist and not all_right_patches_exist:
    #     print("upper-right corner")
    # elif not all_bottom_patches_exist and not all_left_patches_exist:
    #     print("bottom-left corner")
    # elif not all_bottom_patches_exist and not all_right_patches_exist:
    #     print("bottom-right corner")

    border_patches = top_patches + right_patches + bottom_patches + left_patches
    # We have some duplicates because the corners overlap.
    border_patches = list(set(border_patches))

    # We can safely filter nonexistent patches here because we checked if nonexistent
    # patches made sense.
    border_patches = [p for p in border_patches if p.exists()]
    patches_in_roi.extend(border_patches)

    jsons_in_roi = coords_to_paths(
        xs, ys, patch_size=patch_size, extension="json", parent=data_root
    )

    return patches_in_roi, jsons_in_roi


def cv2_add_patchs(
    image: np.ndarray,
    patches: Patches,
    xoff: int = -35917,
    yoff: int = -23945,
    color_negative: ty.Tuple[int, int, int] = (255, 255, 0),
    color_positive: ty.Tuple[int, int, int] = (42, 42, 165),
    line_thickness: int = 5,
) -> np.ndarray:
    """Add patches to image. This adds all squares, not just the exteriors.

    Negative is cyan by default. Positive is brown by default.
    """
    import cv2
    from shapely.affinity import translate

    # tum_patches = [p.polygon for p in patches if p.patch_type == PatchType.TUMOR]
    pos_patches = MultiPolygon(
        [p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.POSITIVE]
    )
    neg_patches = MultiPolygon(
        [p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.NEGATIVE]
    )
    # tum_patches = translate(tum_patches, xoff=xoff, yoff=yoff)
    pos_patches = translate(pos_patches, xoff=xoff, yoff=yoff)
    neg_patches = translate(neg_patches, xoff=xoff, yoff=yoff)

    color_negative = (255, 255, 0)  # cyan (bgr)
    color_positive = (42, 42, 165)  # brown (bgr)

    for geom in pos_patches.geoms:
        poly = np.asarray(list(zip(*geom.exterior.xy))).astype("int32")
        image = cv2.polylines(
            image,
            [poly],
            isClosed=True,
            color=color_positive,
            thickness=line_thickness,
        )

    for geom in neg_patches.geoms:
        poly = np.asarray(list(zip(*geom.exterior.xy))).astype("int32")
        image = cv2.polylines(
            image,
            [poly],
            isClosed=True,
            color=color_negative,
            thickness=line_thickness,
        )

    return image


def cv2_add_patch_exteriors(
    image: np.ndarray,
    patches: Patches,
    xoff: int = -35917,
    yoff: int = -23945,
    color_negative: ty.Tuple[int, int, int] = (255, 255, 0),
    color_positive: ty.Tuple[int, int, int] = (42, 42, 165),
    line_thickness: int = 5,
) -> np.ndarray:
    """Add exterior of patches to an image.

    Negative is cyan by default. Positive is brown by default.
    """
    import cv2
    from shapely.affinity import translate

    tum_patches = [p.polygon for p in patches if p.patch_type == PatchType.TUMOR]
    pos_patches = [
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.POSITIVE
    ]
    neg_patches = [
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.NEGATIVE
    ]

    tum_patches = unary_union(tum_patches)
    pos_patches = unary_union(pos_patches)
    neg_patches = unary_union(neg_patches)

    tum_patches = translate(tum_patches, xoff=xoff, yoff=yoff)
    pos_patches = translate(pos_patches, xoff=xoff, yoff=yoff)
    neg_patches = translate(neg_patches, xoff=xoff, yoff=yoff)
    exts = get_exteriors(tum_patches, pos_patches, neg_patches)

    tumor_exterior: MultiLineString = exts["tumor"]
    tumor_polygon = []
    for line in tumor_exterior:
        multi_tumor_polygon = MultiPolygon(tumor_polygon)
        new_poly = Polygon(line.coords)
        if multi_tumor_polygon.contains(new_poly):
            print("Tumor Polygon found inside another one")
        else:
            tumor_polygon.append(new_poly)
    multi_tumor_polygon = MultiPolygon(tumor_polygon)

    if exts["marker_negative"] is not None:
        for line in exts["marker_negative"]:
            coords = list(zip(*line.xy))
            assert len(coords) == 2
            coords = [(int(x), int(y)) for x, y in coords]
            point1, point2 = Point(coords[0]), Point(coords[1])
            if not (multi_tumor_polygon.contains(point1.buffer(5)) or multi_tumor_polygon.contains(point2.buffer(5))):                
                image = cv2.line(
                    image,
                    coords[0],
                    coords[1],
                    color_negative,
                    thickness=line_thickness,
                )

    if exts["marker_positive"] is not None:
        for line in exts["marker_positive"]:
            coords = list(zip(*line.xy))
            assert len(coords) == 2
            coords = [(int(x), int(y)) for x, y in coords]
            point1, point2 = Point(coords[0]), Point(coords[1])
            if not (multi_tumor_polygon.contains(point1.buffer(5)) or multi_tumor_polygon.contains(point2.buffer(5))):
                image = cv2.line(
                    image,
                    coords[0],
                    coords[1],
                    color_positive,
                    thickness=line_thickness,
                )

    return image


def gen_random_point_per_cell(
    points_data: ty.List[PointOutputData], seed: int = None
) -> ty.Generator[PointOutputData, None, None]:
    """Yield a random point in each cell."""
    random.seed(seed)
    for _, g in itertools.groupby(points_data, lambda p: p.cell_uuid):
        yield random.choice(list(g))


def cv2_add_cell_points(
    image: np.ndarray,
    points_data: ty.Iterable[PointOutputData],
    xoff: int = -35917,
    yoff: int = -23945,
    cell_radius: int = 3,
    color_mapping: ty.Dict[str, ty.Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Add cell points to an image."""
    import cv2
    from shapely.affinity import translate
    import shapely.wkt

    stain_to_color = {
        "cd8": (255, 0, 255),
        "cd16": (0, 255, 255),
        "cd4": (0, 0, 0),
        "cd3": (0, 0, 255),
        "cd163": (0, 255, 0),
    }
    if color_mapping is not None:
        stain_to_color.update(color_mapping)

    for point_data in points_data:
        point = shapely.wkt.loads(point_data.point)
        point = translate(point, xoff=xoff, yoff=yoff)
        point = int(point.x), int(point.y)
        point_color = stain_to_color[point_data.cell_type]
        image = cv2.circle(
            image, center=point, radius=cell_radius, color=point_color, thickness=-1
        )
    return image


def cv2_add_cell_distance_lines(
    image: np.ndarray,
    points_data: ty.Iterable[PointOutputData],
    xoff: int = -35917,
    yoff: int = -23945,
    color_negative: ty.Tuple[int, int, int] = (255, 255, 0),
    color_positive: ty.Tuple[int, int, int] = (42, 42, 165),
    line_thickness: int = 5,
) -> np.ndarray:
    """Add cell distance lines to an image."""
    import cv2
    from shapely.affinity import translate
    import shapely.wkt

    for point_data in points_data:
        if point_data.line_to_marker_pos:
            line_to_pos = shapely.wkt.loads(point_data.line_to_marker_pos)
            line_to_pos = translate(line_to_pos, xoff=xoff, yoff=yoff)
            line_to_pos_start = (
                int(line_to_pos.coords.xy[0][0]),
                int(line_to_pos.coords.xy[1][0]),
            )
            line_to_pos_end = (
                int(line_to_pos.coords.xy[0][1]),
                int(line_to_pos.coords.xy[1][1]),
            )
            image = cv2.line(
                image,
                line_to_pos_start,
                line_to_pos_end,
                color=color_positive,
                thickness=line_thickness,
            )
        if point_data.line_to_marker_neg:
            line_to_neg = shapely.wkt.loads(point_data.line_to_marker_neg)
            line_to_neg = translate(line_to_neg, xoff=xoff, yoff=yoff)
            line_to_neg_start = (
                int(line_to_neg.coords.xy[0][0]),
                int(line_to_neg.coords.xy[1][0]),
            )
            line_to_neg_end = (
                int(line_to_neg.coords.xy[0][1]),
                int(line_to_neg.coords.xy[1][1]),
            )
            image = cv2.line(
                image,
                line_to_neg_start,
                line_to_neg_end,
                color=color_negative,
                thickness=line_thickness,
            )
    return image
