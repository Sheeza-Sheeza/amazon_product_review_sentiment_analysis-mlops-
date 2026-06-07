"""CLI for scoring review text with a trained model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sentiment_analysis.inference.predictor import SentimentPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment for review text.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "best_model" / "model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single review text to classify.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional text file with one review per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts: list[str] = []

    if args.text:
        texts.append(args.text)
    if args.input_file:
        texts.extend(
            line.strip()
            for line in args.input_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not texts:
        raise ValueError("Provide --text or --input-file.")

    predictor = SentimentPredictor(args.model_path)
    results = predictor.predict_batch(texts)
    print(json.dumps(results.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
