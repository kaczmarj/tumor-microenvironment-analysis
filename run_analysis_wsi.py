"""Run distance analysis on an arbitrarily-sized region."""

import argparse
import itertools
import math
import multiprocessing
from pathlib import Path
import time
import typing as ty

import tumor_microenv as tm


def main(
    *,
    data_root: Path,
    output_dir: Path,
    analysis_size: int = 730,
    tumor_microenv: int = 100,
    patch_size: int = 73,
    mpp: float = 0.34622,
    processes: int = None,
):
    # data_root = Path("data_big/N9430-B11/")
    all_patch_paths = list(data_root.glob("*.npy"))
    # Sort by y then by x.
    all_patch_paths.sort(key=lambda p: p.stem.split("_")[:2][::-1])
    assert all_patch_paths, "no patch paths found"

    analysis_size = patch_size * math.ceil(analysis_size / patch_size)
    print(f"Setting analysis_size={analysis_size}")

    def path_to_minx_miny(path):
        path = Path(path)
        splits = path.stem.split("_", maxsplit=4)
        if len(splits) != 4:
            raise ValueError("expected four values in filename")
        minx, miny, _, _ = map(int, splits)
        return minx, miny

    x_y_coords = [path_to_minx_miny(p) for p in all_patch_paths]
    xs = sorted({x for x, _ in x_y_coords})
    ys = sorted({y for _, y in x_y_coords})

    first_minx = xs[0]
    first_miny = ys[0]
    last_minx = xs[-1] + patch_size - analysis_size
    last_miny = ys[-1] + patch_size - analysis_size

    all_xs = list(range(first_minx, last_minx + 1, analysis_size))
    all_ys = list(range(first_miny, last_miny + 1, analysis_size))

    all_xs_ys = list(itertools.product(all_xs, all_ys))
    print(f"maximum number of iterations: {len(all_xs_ys)}")

    def run_one_roi(xy: ty.Tuple[int, int]):
        xmin, ymin = xy
        try:
            patch_paths, cell_paths = tm.get_npy_and_json_files_for_roi(
                xmin=xmin,
                ymin=ymin,
                patch_size=patch_size,
                analysis_size=analysis_size,
                tumor_microenv=round(2 * tumor_microenv / mpp),
                data_root=data_root,
            )
        except tm.CentralPatchFileNotFound:
            # print("some files not found... returning")
            return
        loader = tm.LoaderV1(
            patch_paths,
            cell_paths,
            background=0,
            marker_positive=1,
            marker_negative=7,
        )
        patches, cells = loader()
        cells = [c for c in cells if c.cell_type in {"cd4", "cd8", "cd16", "cd163"}]
        # Get the centroid per cell.
        # cells = [c._replace(polygon=c.polygon.centroid) for c in cells]
        tm.run_spatial_analysis(
            patches=patches,
            cells=cells,
            microenv_distances=[tumor_microenv],
            mpp=mpp,
            output_path=output_dir / f"{xmin}-{ymin}.csv",
            progress_bar=False,
        )

    with multiprocessing.Pool(processes) as pool:
        pool.map(run_one_roi, all_xs_ys)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--analysis-size", type=int, default=730, help="pixels")
    p.add_argument("--tumor-microenv", type=int, default=100, help="micrometers")
    p.add_argument("--patch-size", type=int, default=73, help="pixels")
    p.add_argument("--mpp", type=float, default=0.34622)
    p.add_argument("--processes", type=int, default=None)
    args = p.parse_args()
    t0 = time.time()
    main(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        analysis_size=int(args.analysis_size),
        tumor_microenv=int(args.tumor_microenv),
        patch_size=int(args.patch_size),
        mpp=float(args.mpp),
        processes=int(args.processes),
    )
    t1 = time.time()
    print(f"Finished in {(t1 - t0) / 60:0.1f} minutes")
