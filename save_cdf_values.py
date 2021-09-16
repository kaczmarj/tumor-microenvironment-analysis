"""Plot overlays of tumor boundaries, cells, and distances on top of whole slide."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    points_csvs_root: Path,
    output_dir: Path,
    mpp: float = 0.34622,
):
    print("Reading output CSVs ...")
    df = pd.concat(pd.read_csv(p) for p in points_csvs_root.glob("*.csv"))
    # Save values to make CDF plots.
    save_cdf_values(df=df, mpp=mpp, output_dir=output_dir)


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
    p.add_argument("--points-csvs-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mpp", type=float, default=0.34622)
    args = p.parse_args()
    main(
        points_csvs_root=args.points_csvs_root,
        output_dir=args.output_dir,
        mpp=args.mpp,
    )
