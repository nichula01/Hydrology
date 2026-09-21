#!/usr/bin/env python3
"""
Document and correct negative actual-evapotranspiration values.

The original merged dataset is never modified.

Outputs 
-------
data/processed/srilanka_hydrology_monthly_1982_2011_qc.parquet
data/processed/srilanka_hydrology_monthly_1982_2011_qc.csv.gz

results/tables/negative_aet_records.csv
results/tables/negative_aet_by_year.csv
results/tables/negative_aet_by_month.csv
results/tables/negative_aet_qc_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "srilanka_hydrology_monthly_1982_2011.parquet"
)

OUTPUT_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "srilanka_hydrology_monthly_1982_2011_qc.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "srilanka_hydrology_monthly_1982_2011_qc.csv.gz"
)

TABLE_DIR = ROOT / "results" / "tables"

AET_COLUMN = "actual_evapotranspiration_mm"
RAW_AET_COLUMN = "actual_evapotranspiration_mm_raw"
FLAG_COLUMN = "aet_negative_corrected"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset does not exist: {INPUT_PATH}"
        )

    frame = pd.read_parquet(INPUT_PATH)

    negative_mask = frame[AET_COLUMN] < 0
    negative_records = frame.loc[
        negative_mask,
        [
            "grid_id",
            "longitude",
            "latitude",
            "date",
            "year",
            "month",
            "precipitation_mm",
            "PET_mm",
            AET_COLUMN,
            "total_runoff_mm",
            "soil_moisture_total_mm",
        ],
    ].copy()

    negative_count = int(negative_mask.sum())
    total_count = int(len(frame))
    negative_percentage = 100.0 * negative_count / total_count

    if negative_count == 0:
        minimum_negative = None
        maximum_negative = None
        mean_negative = None
        median_negative = None
    else:
        values = negative_records[AET_COLUMN]
        minimum_negative = float(values.min())
        maximum_negative = float(values.max())
        mean_negative = float(values.mean())
        median_negative = float(values.median())

    # Preserve the unmodified AET values.
    frame[RAW_AET_COLUMN] = frame[AET_COLUMN]

    # Flag records corrected by the physical non-negativity rule.
    frame[FLAG_COLUMN] = negative_mask

    # Apply the correction only to the working AET variable.
    frame[AET_COLUMN] = frame[AET_COLUMN].clip(lower=0.0)

    remaining_negative_count = int(
        (frame[AET_COLUMN] < 0).sum()
    )

    corrected_difference = (
        frame[AET_COLUMN] - frame[RAW_AET_COLUMN]
    )

    negative_records = negative_records.sort_values(
        ["year", "month", "latitude", "longitude"]
    ).reset_index(drop=True)

    by_year = (
        negative_records.groupby("year", as_index=False)
        .agg(
            negative_record_count=(AET_COLUMN, "size"),
            minimum_negative_aet_mm=(AET_COLUMN, "min"),
            mean_negative_aet_mm=(AET_COLUMN, "mean"),
            maximum_negative_aet_mm=(AET_COLUMN, "max"),
        )
        .sort_values("year")
    )

    by_month = (
        negative_records.groupby("month", as_index=False)
        .agg(
            negative_record_count=(AET_COLUMN, "size"),
            minimum_negative_aet_mm=(AET_COLUMN, "min"),
            mean_negative_aet_mm=(AET_COLUMN, "mean"),
            maximum_negative_aet_mm=(AET_COLUMN, "max"),
        )
        .sort_values("month")
    )

    negative_records.to_csv(
        TABLE_DIR / "negative_aet_records.csv",
        index=False,
    )

    by_year.to_csv(
        TABLE_DIR / "negative_aet_by_year.csv",
        index=False,
    )

    by_month.to_csv(
        TABLE_DIR / "negative_aet_by_month.csv",
        index=False,
    )

    frame.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        compression="snappy",
    )

    frame.to_csv(
        OUTPUT_CSV,
        index=False,
        compression="gzip",
    )

    summary = {
        "status": (
            "PASS"
            if remaining_negative_count == 0
            else "FAIL"
        ),
        "input_dataset": str(INPUT_PATH),
        "output_parquet": str(OUTPUT_PARQUET),
        "output_compressed_csv": str(OUTPUT_CSV),
        "total_records": total_count,
        "negative_aet_records_before_correction": negative_count,
        "negative_aet_percentage": negative_percentage,
        "minimum_negative_aet_mm": minimum_negative,
        "maximum_negative_aet_mm": maximum_negative,
        "mean_negative_aet_mm": mean_negative,
        "median_negative_aet_mm": median_negative,
        "remaining_negative_aet_records": (
            remaining_negative_count
        ),
        "maximum_absolute_correction_mm": float(
            corrected_difference.abs().max()
        ),
        "mean_absolute_correction_all_records_mm": float(
            corrected_difference.abs().mean()
        ),
        "correction_rule": (
            "actual_evapotranspiration_mm = "
            "max(actual_evapotranspiration_mm_raw, 0)"
        ),
        "raw_values_preserved_in_column": RAW_AET_COLUMN,
        "corrected_records_flagged_in_column": FLAG_COLUMN,
    }

    summary_path = (
        TABLE_DIR / "negative_aet_qc_summary.json"
    )

    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print("=" * 76)
    print("NEGATIVE ACTUAL-EVAPOTRANSPIRATION QUALITY CONTROL")
    print("=" * 76)
    print(f"Status                         : {summary['status']}")
    print(f"Total records                  : {total_count:,}")
    print(f"Negative AET records           : {negative_count}")
    print(
        f"Percentage of dataset          : "
        f"{negative_percentage:.6f}%"
    )
    print(
        f"Minimum negative AET           : "
        f"{minimum_negative:.8f} mm"
    )
    print(
        f"Mean negative AET              : "
        f"{mean_negative:.8f} mm"
    )
    print(
        f"Maximum negative AET           : "
        f"{maximum_negative:.8f} mm"
    )
    print(
        f"Remaining negative values      : "
        f"{remaining_negative_count}"
    )
    print(
        f"Maximum absolute correction    : "
        f"{summary['maximum_absolute_correction_mm']:.8f} mm"
    )
    print()
    print(f"QC Parquet dataset             : {OUTPUT_PARQUET}")
    print(f"QC compressed CSV              : {OUTPUT_CSV}")
    print(
        f"Negative-record table          : "
        f"{TABLE_DIR / 'negative_aet_records.csv'}"
    )
    print(
        f"Year summary                   : "
        f"{TABLE_DIR / 'negative_aet_by_year.csv'}"
    )
    print(
        f"Month summary                  : "
        f"{TABLE_DIR / 'negative_aet_by_month.csv'}"
    )
    print(f"QC summary                     : {summary_path}")
    print("=" * 76)

    if remaining_negative_count != 0:
        raise SystemExit(
            "Negative AET values remain after correction."
        )


if __name__ == "__main__":
    main()
