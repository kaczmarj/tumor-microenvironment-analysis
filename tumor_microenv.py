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
    polygon: Polygon
    patch_type: PatchType
    biomarker_status: BiomarkerStatus


class Cell(ty.NamedTuple):
    polygon: Polygon
    cell_type: str
    uuid: str

    @property
    def lattice_points(self) -> MultiPoint:
        """Lattice points of the polygon (all integer points that make up polygon)."""
        # convert values to int
        xmin, ymin, xmax, ymax = map(round, self.polygon.bounds)
        coords = itertools.product(range(xmin, xmax + 1), range(ymin, ymax + 1))
        points = (Point(*p) for p in coords)
        points = MultiPoint(list(points)).intersection(self.polygon)
        return points


Patches = ty.List[Patch]
Cells = ty.List[Cell]


class PosNegDistances(ty.NamedTuple):
    dpositive: float
    dnegative: float


class PosNegLines(ty.NamedTuple):
    line_to_positive: LineString
    line_to_negative: LineString


class PointOutputData(ty.NamedTuple):
    """Object representing one row in the output file."""

    point: str
    dist_to_marker_neg: float
    dist_to_marker_pos: float
    line_to_marker_neg: str
    line_to_marker_pos: str
    cell_type: str
    cell_uuid: str
    microenv_micrometer: int


def _get_distances_for_point(
    point: Point,
    positive_patches: _base_geometry.BaseGeometry,
    negative_patches: _base_geometry.BaseGeometry,
) -> PosNegDistances:
    """Get the distances from the point to the nearest positive and negative patches."""
    dpos = point.distance(positive_patches)
    dneg = point.distance(negative_patches)
    return PosNegDistances(dpositive=dpos, dnegative=dneg)


def _get_nearest_points_for_point(
    point: Point,
    positive_patches: _base_geometry.BaseGeometry,
    negative_patches: _base_geometry.BaseGeometry,
) -> PosNegLines:
    """Get the lines joining the point to the nearest positive patch and the nearest
    negative patch.
    """
    line_to_pos = nearest_points(point, positive_patches)
    line_to_neg = nearest_points(point, negative_patches)
    line_to_pos = LineString(line_to_pos)
    line_to_neg = LineString(line_to_neg)
    return PosNegLines(line_to_positive=line_to_pos, line_to_negative=line_to_neg)


def _get_exterior_of_multigeom(
    multigeom: _base_geometry.BaseMultipartGeometry,
) -> MultiLineString:
    return MultiLineString([g.exterior for g in multigeom.geoms])


def _exterior_to_multilinestring(
    exterior: _base_geometry.BaseGeometry,
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
    exterior = _get_exterior_of_multigeom(multigeom)
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
    for cell_point in cell.lattice_points.geoms:
        distances = _get_distances_for_point(
            cell_point,
            positive_patches=marker_positive_geom,
            negative_patches=marker_negative_geom,
        )
        try:
            lines_from_point_to_patches = _get_nearest_points_for_point(
                cell_point,
                positive_patches=marker_positive_geom,
                negative_patches=marker_negative_geom,
            )
        # Sometimes this can error... but why?
        except ValueError:
            continue
        yield PointOutputData(
            point=cell_point.wkt,
            dist_to_marker_neg=distances.dnegative,
            dist_to_marker_pos=distances.dpositive,
            line_to_marker_neg=lines_from_point_to_patches.line_to_negative.wkt,
            line_to_marker_pos=lines_from_point_to_patches.line_to_positive.wkt,
            cell_type=cell.cell_type,
            cell_uuid=cell.uuid,
            microenv_micrometer=microenv_micrometer,
        )


def get_exteriors(
    tumor: _base_geometry.BaseMultipartGeometry,
    biomarker_positive: _base_geometry.BaseGeometry,
    biomarker_negative: _base_geometry.BaseGeometry,
) -> ty.Dict[str, MultiLineString]:
    """Get exteriors of tumor, biomarker-positive, and biomarker-negative polygons."""
    tumor_exterior = _get_exterior_of_multigeom(tumor)
    marker_positive_exterior = _get_exterior_contained_in_larger_geom(
        biomarker_positive, tumor_exterior
    )
    marker_negative_exterior = _get_exterior_contained_in_larger_geom(
        biomarker_negative, tumor_exterior
    )
    return dict(
        tumor=tumor_exterior,
        marker_positive=marker_positive_exterior,
        marker_negative=marker_negative_exterior,
    )


def run_spatial_analysis(
    patches: Patches,
    cells: Cells,
    microenv_distances: ty.Sequence[int],
    mpp: float,
    output_path: PathType = "output.csv",
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
    """
    # This is a multipolygon that represents the entire tumor in our region of interest.
    tumor_geom: MultiPolygon = unary_union(
        [p.polygon for p in patches if p.patch_type == PatchType.TUMOR]
    )
    marker_positive_patches = (
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.POSITIVE
    )
    marker_negative_patches = (
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.NEGATIVE
    )
    marker_positive_geom = unary_union(list(marker_positive_patches))
    marker_negative_geom = unary_union(list(marker_negative_patches))

    exteriors = get_exteriors(
        tumor=tumor_geom,
        biomarker_positive=marker_positive_geom,
        biomarker_negative=marker_negative_geom,
    )
    tumor_exterior = exteriors["tumor"]
    marker_positive_geom = exteriors["marker_positive"]
    marker_negative_geom = exteriors["marker_negative"]
    del marker_positive_patches, marker_negative_patches

    # Polygons of blank tiles, so we can exclude cells in these regions.
    # blank_tiles = unary_union(
    #     [p.polygon for p in patches if p.patch_type == PatchType.BLANK]
    # )

    with open(output_path, "w", newline="") as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=PointOutputData._fields)
        dict_writer.writeheader()

        for distance_um in microenv_distances:
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
            print("Filtering cells...")
            cells_in_microenv = [
                cell
                for cell in cells
                if tumor_microenv.contains(cell.polygon)
                # and not blank_tiles.intersects(cell.polygon)
            ]
            for cell in tqdm(cells_in_microenv):
                row_generator = _distances_for_cell_in_microenv(
                    cell=cell,
                    marker_positive_geom=marker_positive_geom,
                    marker_negative_geom=marker_negative_geom,
                    microenv_micrometer=distance_um,
                )
                for row in row_generator:
                    # Only write the data if it is within our microenvironment.
                    # max_dist = max(row.dist_to_marker_pos, row.dist_to_marker_neg)
                    # if max_dist <= distance_px:
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
            raise ValueError("expected self.load() to return two objects")
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
    ):
        self.patch_paths = patch_paths
        self.cells_json = cells_json
        self.background = background
        self.marker_positive = marker_positive
        self.marker_negative = marker_negative

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
        if percent_tumor >= 0.05:
            n_tumor_points = tumor_mask.sum()
            if marker_pos_mask.sum() / n_tumor_points >= 0.40:
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


def overlay_patches_and_points(
    image_path: PathType,
    patches: Patches,
    points_data: PointOutputData,
    xoff: int = -35917,
    yoff: int = -23945,
    output_path: PathType = "overlay.png",
):
    import cv2
    from shapely.affinity import translate
    import shapely.wkt

    color_negative = (255, 255, 0)  # cyan (bgr)
    color_positive = (42, 42, 165)  # brown (bgr)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"error loading image: {image_path}")
    marker_positive_patches = (
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.POSITIVE
    )
    marker_negative_patches = (
        p.polygon for p in patches if p.biomarker_status == BiomarkerStatus.NEGATIVE
    )
    marker_positive_geom = unary_union(list(marker_positive_patches))
    marker_negative_geom = unary_union(list(marker_negative_patches))
    # Translate so we are in same coordinate space as image.
    marker_positive_geom = translate(marker_positive_geom, xoff=xoff, yoff=yoff)
    marker_negative_geom = translate(marker_negative_geom, xoff=xoff, yoff=yoff)

    for geom in marker_positive_geom.geoms:
        poly = np.asarray(list(zip(*geom.exterior.xy))).astype("int32")
        image = cv2.polylines(
            image,
            [poly],
            isClosed=True,
            color=color_positive,
            thickness=5,
        )

    for geom in marker_negative_geom.geoms:
        poly = np.asarray(list(zip(*geom.exterior.xy))).astype("int32")
        image = cv2.polylines(
            image,
            [poly],
            isClosed=True,
            color=color_negative,
            thickness=5,
        )

    def gen_random_point_per_cell(points_data):
        for _, g in itertools.groupby(points_data, lambda p: p.cell_uuid):
            yield random.choice(list(g))

    random_points_per_cell = list(gen_random_point_per_cell(points_data))
    stain_to_color = {
        "cd8": (255, 0, 255),
        "cd16": (0, 255, 255),
        "cd4": (0, 0, 0),
        "cd3": (0, 0, 255),
        "cd163": (0, 255, 0),
    }

    for point_data in random_points_per_cell:
        point = shapely.wkt.loads(point_data.point)
        point = translate(point, xoff=xoff, yoff=yoff)
        point = int(point.x), int(point.y)
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
        point_color = stain_to_color[point_data.cell_type]
        image = cv2.circle(image, point, 3, point_color, -1)
        image = cv2.line(
            image, line_to_pos_start, line_to_pos_end, color=color_positive, thickness=1
        )
        image = cv2.line(
            image, line_to_neg_start, line_to_neg_end, color=color_negative, thickness=1
        )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"error saving image to {output_path}")
