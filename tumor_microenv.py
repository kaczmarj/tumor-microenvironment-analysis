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
import typing as ty
import uuid

import numpy as np
from shapely.geometry import box as box_constructor
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.ops import unary_union
from tqdm import tqdm

PathType = ty.Union[str, bytes, Path]


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


def _get_tumor_microenvironment(
    tumor_geom: MultiPolygon, distance_px: int
) -> MultiPolygon:
    """Return a dilated MultiPolygon of tumors, representing the microenvironment at a
    given distance (in pixels).
    """
    return tumor_geom.buffer(distance=distance_px)


def _get_distances_for_point(
    point: Point, positive_patches, negative_patches
) -> PosNegDistances:
    """Get the distances from the point to the nearest positive and negative patches."""
    dpos = point.distance(positive_patches)
    dneg = point.distance(negative_patches)
    return PosNegDistances(dpositive=dpos, dnegative=dneg)


def _get_nearest_points_for_point(
    point: Point, positive_patches, negative_patches
) -> PosNegLines:
    """Get the lines joining the point to the nearest positive patch and the nearest
    negative patch.
    """
    line_to_pos = nearest_points(point, positive_patches)
    line_to_neg = nearest_points(point, negative_patches)
    line_to_pos = LineString(line_to_pos)
    line_to_neg = LineString(line_to_neg)
    return PosNegLines(line_to_positive=line_to_pos, line_to_negative=line_to_neg)


def _distances_for_cell_in_microenv(
    cell: Cell,
    marker_positive_geom,
    marker_negative_geom,
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
    del marker_positive_patches, marker_negative_patches

    # Polygons of blank tiles, so we can exclude cells in these regions.
    blank_tiles = unary_union(
        [p.polygon for p in patches if p.patch_type == PatchType.BLANK]
    )

    with open(output_path, "w", newline="") as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=PointOutputData._fields)
        dict_writer.writeheader()

        for distance_um in microenv_distances:
            distance_px = round(distance_um / mpp)
            print(f"Working on distance = {distance_um} um ({distance_px} px)")
            tumor_microenv = _get_tumor_microenvironment(
                tumor_geom=tumor_geom, distance_px=distance_px
            )

            # TODO: this is NOT the same as the method we discussed with Joel and
            # Mahmudul. We discussed getting all of the POINTS inside the
            # microenvironment. But here, we take all of the CELLS in the
            # microenvironment. It's much easier to implement this, so let's roll with
            # it.
            print(
                "Finding cells that are inside the tumor microenvironment and not in"
                " blank tiles ..."
            )
            cells_in_microenv = [
                cell
                for cell in cells
                if tumor_microenv.contains(cell.polygon)
                and not blank_tiles.intersects(cell.polygon)
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
                    max_dist = max(row.dist_to_marker_pos, row.dist_to_marker_neg)
                    if max_dist <= distance_px:
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
        marker_neg_thresh: float = 0.3,
    ):
        self.patch_paths = patch_paths
        self.cells_json = cells_json
        self.background = background
        self.marker_positive = marker_positive
        self.marker_negative = marker_negative
        self.marker_neg_thresh = marker_neg_thresh

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
        blank_mask = patch == self.background
        percent_tumor = tumor_mask.mean()
        percent_blank = blank_mask.mean()
        if percent_tumor > 0.5:
            n_tumor_points = tumor_mask.sum()
            if marker_neg_mask.sum() / n_tumor_points > self.marker_neg_thresh:
                return PatchType.TUMOR, BiomarkerStatus.NEGATIVE
            else:
                return PatchType.TUMOR, BiomarkerStatus.POSITIVE
        elif percent_blank > 0.5:
            return PatchType.BLANK, BiomarkerStatus.NA
        else:
            return PatchType.NONTUMOR, BiomarkerStatus.NA

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


def plot_point_data_and_tumor(
    patches: Patches, points_data: ty.Sequence[PointOutputData]
):
    """Instead of being an extensible plotting function, this is implemented as an
    example of plotting our data.
    """
    import matplotlib.pyplot as plt
    import shapely.wkt

    colors = {
        BiomarkerStatus.NA: "white",
        BiomarkerStatus.POSITIVE: "brown",
        BiomarkerStatus.NEGATIVE: "cyan",
    }
    for patch in patches:
        # Plot tumor patches.
        if patch.patch_type == PatchType.TUMOR:
            plt.fill(*patch.polygon.exterior.xy, color=colors[patch.biomarker_status])
        # Plot the point we are interested in.
    for point_data in points_data:
        point = shapely.wkt.loads(point_data.point)
        plt.plot(point.x, point.y, color="black", marker="o")
        # Plot line to the nearest marker-positive region.
        line_to_pos = shapely.wkt.loads(point_data.line_to_marker_pos)
        plt.plot(*line_to_pos.xy, color="brown", linestyle="--")
        # Plot line to the nearest marker-negative region.
        line_to_neg = shapely.wkt.loads(point_data.line_to_marker_neg)
        plt.plot(*line_to_neg.xy, color="cyan", linestyle="--")
