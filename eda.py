"""
Exploratory Data Analysis (EDA) for Amazon review sentiment dataset.

Run:
    python eda.py
    python eda.py --data-path path/to/dataset.csv --output-dir .
"""

from __future__ import annotations

import argparse
from datetime import datetime
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TEXT_COLUMN_CANDIDATES = (
    "reviews.text",
    "reviewText",
    "review_text",
    "text",
    "review",
    "content",
    "body",
)

LABEL_COLUMN_CANDIDATES = (
    "reviews.rating",
    "rating",
    "label",
    "sentiment",
    "class",
    "target",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on a review dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("7817_1.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where eda_report.txt and plots are saved.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Override auto-detected text column name.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Override auto-detected label/rating column name.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV with fallback encodings for common encoding issues."""
    encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Loaded dataset with encoding: {encoding}")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Failed to load {path} with encodings {encodings}") from last_error


def detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def print_basic_info(df: pd.DataFrame) -> None:
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    print("\nData types:")
    print(df.dtypes.to_string())


def run_data_quality_checks(df: pd.DataFrame) -> dict[str, object]:
    print("\n=== Data Quality Checks ===")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )

    print("\nMissing values per column (non-zero only):")
    if missing_df.empty:
        print("  No missing values found.")
    else:
        print(missing_df.to_string())

    duplicate_count = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count}")

    return {
        "missing_values": missing_df,
        "duplicate_rows": duplicate_count,
    }


def run_basic_statistics(df: pd.DataFrame) -> dict[str, object]:
    print("\n=== Basic Statistics ===")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_summary = None
    if numeric_cols:
        print("\nNumerical columns - describe():")
        numeric_summary = df[numeric_cols].describe()
        print(numeric_summary.to_string())
    else:
        print("\nNo numerical columns found.")

    categorical_summaries: dict[str, pd.Series] = {}
    if categorical_cols:
        print("\nCategorical columns - value_counts():")
        for col in categorical_cols:
            counts = df[col].value_counts(dropna=False)
            categorical_summaries[col] = counts
            print(f"\n[{col}] (top 10):")
            print(counts.head(10).to_string())
    else:
        print("\nNo categorical columns found.")

    return {
        "numeric_summary": numeric_summary,
        "categorical_summaries": categorical_summaries,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def analyze_text_column(df: pd.DataFrame, text_col: str) -> dict[str, object]:
    print(f"\n=== Text Column Analysis: {text_col} ===")

    text_series = df[text_col].fillna("").astype(str)
    non_empty = text_series.str.strip().ne("")

    samples = df.loc[non_empty, text_col].sample(
        n=min(5, non_empty.sum()),
        random_state=42,
    )

    print("\n5 random text samples:")
    for idx, (row_idx, text) in enumerate(samples.items(), start=1):
        preview = str(text).replace("\n", " ")[:300]
        print(f"\n  Sample {idx} (row {row_idx}):")
        print(f"    {preview}{'...' if len(str(text)) > 300 else ''}")

    lengths = text_series.str.len()
    avg_length = lengths.mean()
    median_length = lengths.median()
    empty_count = int((~non_empty).sum())

    print(f"\nAverage text length: {avg_length:.2f} characters")
    print(f"Median text length: {median_length:.2f} characters")
    print(f"Empty/missing text rows: {empty_count}")

    return {
        "text_column": text_col,
        "avg_text_length": avg_length,
        "median_text_length": median_length,
        "empty_text_count": empty_count,
        "samples": samples,
    }


def analyze_label_column(
    df: pd.DataFrame,
    label_col: str,
    output_dir: Path,
) -> dict[str, object]:
    print(f"\n=== Label/Rating Analysis: {label_col} ===")

    labels = df[label_col].dropna()
    class_counts = labels.value_counts().sort_index()
    total = int(class_counts.sum())
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    imbalance_ratio = class_counts.max() / class_counts.min()

    print("\nClass distribution:")
    print(class_counts.to_string())
    print(f"\nClass imbalance ratio (majority/minority): {imbalance_ratio:.2f}")
    print(f"Majority class: {majority_class} ({class_counts.max()} samples)")
    print(f"Minority class: {minority_class} ({class_counts.min()} samples)")

    plot_path = output_dir / "label_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(f"Distribution of {label_col}")
    ax.set_xlabel(label_col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    for idx, count in enumerate(class_counts.values):
        ax.text(idx, count, str(count), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved label distribution plot: {plot_path}")

    return {
        "label_column": label_col,
        "class_counts": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "plot_path": plot_path,
        "labeled_rows": total,
        "missing_labels": int(df[label_col].isnull().sum()),
    }


def build_report(
    df: pd.DataFrame,
    quality: dict[str, object],
    stats: dict[str, object],
    text_analysis: dict[str, object] | None,
    label_analysis: dict[str, object] | None,
    data_path: Path,
) -> str:
    buffer = StringIO()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buffer.write("=" * 60 + "\n")
    buffer.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    buffer.write("=" * 60 + "\n\n")
    buffer.write(f"Generated at : {timestamp}\n")
    buffer.write(f"Dataset path : {data_path.resolve()}\n")
    buffer.write(f"Rows x Cols  : {df.shape[0]} x {df.shape[1]}\n\n")

    buffer.write("-" * 60 + "\n")
    buffer.write("1. COLUMN SUMMARY\n")
    buffer.write("-" * 60 + "\n")
    buffer.write("Columns:\n")
    for col in df.columns:
        buffer.write(f"  - {col} ({df[col].dtype})\n")
    buffer.write("\n")

    buffer.write("-" * 60 + "\n")
    buffer.write("2. DATA QUALITY\n")
    buffer.write("-" * 60 + "\n")
    buffer.write(f"Duplicate rows: {quality['duplicate_rows']}\n\n")
    missing_df: pd.DataFrame = quality["missing_values"]
    if missing_df.empty:
        buffer.write("Missing values: none\n\n")
    else:
        buffer.write("Missing values (non-zero columns):\n")
        buffer.write(missing_df.to_string())
        buffer.write("\n\n")

    buffer.write("-" * 60 + "\n")
    buffer.write("3. NUMERICAL STATISTICS\n")
    buffer.write("-" * 60 + "\n")
    numeric_summary: pd.DataFrame | None = stats["numeric_summary"]
    if numeric_summary is not None:
        buffer.write(numeric_summary.to_string())
        buffer.write("\n\n")
    else:
        buffer.write("No numerical columns.\n\n")

    buffer.write("-" * 60 + "\n")
    buffer.write("4. CATEGORICAL STATISTICS (TOP VALUES)\n")
    buffer.write("-" * 60 + "\n")
    categorical_summaries: dict[str, pd.Series] = stats["categorical_summaries"]
    if not categorical_summaries:
        buffer.write("No categorical columns.\n\n")
    else:
        for col, counts in categorical_summaries.items():
            buffer.write(f"\n[{col}] top 10:\n")
            buffer.write(counts.head(10).to_string())
            buffer.write("\n")
        buffer.write("\n")

    if text_analysis:
        buffer.write("-" * 60 + "\n")
        buffer.write("5. TEXT ANALYSIS\n")
        buffer.write("-" * 60 + "\n")
        buffer.write(f"Text column         : {text_analysis['text_column']}\n")
        buffer.write(f"Average text length : {text_analysis['avg_text_length']:.2f}\n")
        buffer.write(f"Median text length  : {text_analysis['median_text_length']:.2f}\n")
        buffer.write(f"Empty text rows     : {text_analysis['empty_text_count']}\n\n")
        buffer.write("Random samples:\n")
        for idx, (row_idx, text) in enumerate(text_analysis["samples"].items(), start=1):
            preview = str(text).replace("\n", " ")[:250]
            buffer.write(f"  [{idx}] row {row_idx}: {preview}\n")
        buffer.write("\n")

    if label_analysis:
        buffer.write("-" * 60 + "\n")
        buffer.write("6. LABEL / RATING ANALYSIS\n")
        buffer.write("-" * 60 + "\n")
        buffer.write(f"Label column        : {label_analysis['label_column']}\n")
        buffer.write(f"Labeled rows        : {label_analysis['labeled_rows']}\n")
        buffer.write(f"Missing labels      : {label_analysis['missing_labels']}\n")
        buffer.write(
            f"Imbalance ratio     : {label_analysis['imbalance_ratio']:.2f}\n"
        )
        buffer.write(
            f"Majority class      : {label_analysis['majority_class']}\n"
        )
        buffer.write(
            f"Minority class      : {label_analysis['minority_class']}\n\n"
        )
        buffer.write("Class counts:\n")
        buffer.write(label_analysis["class_counts"].to_string())
        buffer.write("\n\n")
        buffer.write(f"Plot saved to: {label_analysis['plot_path']}\n\n")

    buffer.write("-" * 60 + "\n")
    buffer.write("7. KEY FINDINGS\n")
    buffer.write("-" * 60 + "\n")
    buffer.write(f"- Dataset contains {df.shape[0]} records and {df.shape[1]} features.\n")
    buffer.write(f"- Found {quality['duplicate_rows']} duplicate rows.\n")

    if not missing_df.empty:
        top_missing = missing_df.iloc[0]
        buffer.write(
            f"- Highest missingness: {top_missing.name} "
            f"({int(top_missing['missing_count'])} rows, {top_missing['missing_pct']}%)\n"
        )

    if text_analysis:
        buffer.write(
            f"- Review text avg length is {text_analysis['avg_text_length']:.0f} characters.\n"
        )

    if label_analysis:
        buffer.write(
            f"- Rating distribution is imbalanced (ratio {label_analysis['imbalance_ratio']:.2f}:1).\n"
        )
        buffer.write(
            f"- Most frequent rating: {label_analysis['majority_class']}; "
            f"least frequent: {label_analysis['minority_class']}.\n"
        )

    buffer.write("\nEnd of report.\n")
    return buffer.getvalue()


def save_report(report: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eda_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    data_path = args.data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_dataset(data_path)

    print_basic_info(df)
    quality = run_data_quality_checks(df)
    stats = run_basic_statistics(df)

    text_col = args.text_column or detect_column(df.columns.tolist(), TEXT_COLUMN_CANDIDATES)
    label_col = args.label_column or detect_column(df.columns.tolist(), LABEL_COLUMN_CANDIDATES)

    text_analysis = None
    if text_col:
        text_analysis = analyze_text_column(df, text_col)
    else:
        print("\nNo text column detected. Skipping text analysis.")

    label_analysis = None
    if label_col:
        label_analysis = analyze_label_column(df, label_col, output_dir)
    else:
        print("\nNo label/rating column detected. Skipping label analysis.")

    report = build_report(
        df=df,
        quality=quality,
        stats=stats,
        text_analysis=text_analysis,
        label_analysis=label_analysis,
        data_path=data_path,
    )
    report_path = save_report(report, output_dir)

    print("\n=== EDA Complete ===")
    print(f"Report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
