"""Plot overlays of tumor boundaries, cells, and distances on top of whole slide."""

import argparse
import itertools
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd

import tumor_microenv as tm


def main(
    roi_path: Path,
    data_root: Path,
    points_csvs_root: Path,
    output_dir: Path,
    patch_size: int = 73,
    mpp: float = 0.34622,
):
    print("Reading output CSVs ...")
    df = pd.concat(pd.read_csv(p) for p in points_csvs_root.glob("*.csv"))
    # Save values to make CDF plots.
    save_cdf_values(df=df, mpp=mpp, output_dir=output_dir)
    with tempfile.NamedTemporaryFile() as f:
        df.to_csv(f.name, index=False)
        points_data = tm.read_point_csv(f.name)
    del df

    print("Reading image ...")
    offset_path = roi_path.parent / "offset.txt"
    assert roi_path.exists(), f"file not found: {roi_path}"
    assert offset_path.exists(), f"file not found: {offset_path}"

    xoff, yoff = map(int, offset_path.read_text().split())
    print(f"  offsets: x={xoff}, y={yoff}")

    image: np.ndarray = cv2.imread(str(roi_path))
    assert image is not None, f"error loading {roi_path}"

    cols, rows = image.shape[:2]
    last_x = xoff + cols - patch_size
    last_y = yoff + rows - patch_size

    # Get all of the x, y coordinates for the relevant patches.
    xs = list(range(xoff, last_x + 1, patch_size))
    ys = list(range(yoff, last_y + 1, patch_size))
    coords = itertools.product(xs, ys)
    patch_paths = [
        data_root / f"{x}_{y}_{patch_size}_{patch_size}.npy" for x, y in coords
    ]

    print("Loading patches ...")
    patches, _ = tm.LoaderV1(
        patch_paths, [], background=0, marker_positive=1, marker_negative=7
    )()

    print("Creating image with tumor exteriors only")
    image = tm.cv2_add_patch_exteriors(
        image, patches=patches, xoff=-xoff, yoff=-yoff, line_thickness=10
    )
    print("Saving image with tumor exteriors only")
    assert cv2.imwrite(str(output_dir / "overlay-boundaries.png"), image)

    image = tm.cv2_add_cell_distance_lines(
        image,
        points_data=points_data,
        xoff=-xoff,
        yoff=-yoff,
        line_thickness=2,
    )
    image = tm.cv2_add_cell_points(
        image, points_data=points_data, xoff=-xoff, yoff=-yoff, cell_radius=15
    )
    print("Saving image with tumor exteriors, cells, and distance lines")
    assert cv2.imwrite(str(output_dir / "overlay-boundaries-cells-lines.png"), image)


def save_cdf_values(df: pd.DataFrame, mpp: float, output_dir: Path):
    df.loc[:, "dist_to_marker_neg"] *= mpp
    df.loc[:, "dist_to_marker_pos"] *= mpp

    def get_pdf_and_cdf(values, bins=100):
        values = np.asarray(values)
        count, bins = np.histogram(values, bins=100)
        pdf = count / count.sum()
        cdf = pdf.cumsum()
        return pdf, cdf, bins

    cell_types = ["cd4", "cd8", "cd16", "cd163"]
    for cell_type in cell_types:
        pdf_neg, cdf_neg, bins_neg = get_pdf_and_cdf(
            df.query(f"cell_type=='{cell_type}'").loc[:, "dist_to_marker_neg"]
        )
        pdf_pos, cdf_pos, bins_pos = get_pdf_and_cdf(
            df.query(f"cell_type=='{cell_type}'").loc[:, "dist_to_marker_pos"]
        )
        f = output_dir / f"cdf_values_{cell_type}.npz"
        print(f"Saving file {f}")
        np.savez_compressed(
            f,
            pdf_neg=pdf_neg,
            cdf_neg=cdf_neg,
            bins_neg=bins_neg,
            pdf_pos=pdf_pos,
            cdf_pos=cdf_pos,
            bins_pos=bins_pos,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roi-path", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True, help="path to npy files")
    p.add_argument("--points-csvs-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--patch-size", type=int, default=73)
    p.add_argument("--mpp", type=float, default=0.34622)
    args = p.parse_args()
    main(
        roi_path=args.roi_path,
        data_root=args.data_root,
        points_csvs_root=args.points_csvs_root,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
    )
