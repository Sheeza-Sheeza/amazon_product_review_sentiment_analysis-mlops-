"""DVC stage: prepare dataset for training."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sentiment_analysis.data.loader import load_review_dataset  # noqa: E402
from sentiment_analysis.data.preprocessing import prepare_training_data  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "7817_1.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews.parquet"


def main() -> None:
    if not RAW_PATH.exists():
        fallback = PROJECT_ROOT / "7817_1.csv"
        if fallback.exists():
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            RAW_PATH.write_bytes(fallback.read_bytes())

    df, text_col, rating_col = load_review_dataset(RAW_PATH)
    prepared = prepare_training_data(df, text_col, rating_col)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(prepared)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
