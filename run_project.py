"""Automatic runner for the MBTI prediction project.

Default mode loads the saved dichotomy classifier and evaluates it on the
processed test split. Use --train to retrain from the processed train split
before evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.dichotomy_classifiers import DichotomyClassifiers


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE = (
    "I love deep philosophical discussions and spending time alone with my thoughts."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MBTI dichotomy classifier evaluation."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain the dichotomy classifier before evaluation.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing train.csv.gz and test.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing or receiving dichotomy_classifiers.pkl.",
    )
    parser.add_argument(
        "--sample",
        default=DEFAULT_SAMPLE,
        help="Sample text to classify after evaluation.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_test_data(data_dir: Path) -> pd.DataFrame:
    test_path = data_dir / "test.csv"
    require_file(test_path)
    return pd.read_csv(test_path)


def train_model(data_dir: Path, results_dir: Path) -> DichotomyClassifiers:
    train_path = data_dir / "train.csv.gz"
    require_file(train_path)

    print(f"Loading training data: {train_path}")
    df_train = pd.read_csv(train_path, compression="infer")

    print("Training dichotomy classifier...")
    clf = DichotomyClassifiers()
    clf.fit_from_df(df_train)
    clf.save(results_dir)
    return clf


def load_model(results_dir: Path) -> DichotomyClassifiers:
    model_path = results_dir / "dichotomy_classifiers.pkl"
    require_file(model_path)

    clf = DichotomyClassifiers()
    clf.load(results_dir)
    return clf


def evaluate_model(clf: DichotomyClassifiers, df_test: pd.DataFrame) -> None:
    print(f"Evaluating on {len(df_test)} test examples...")
    results = clf.evaluate_from_df(df_test)

    print("\nEvaluation summary")
    for name, metrics in results.items():
        if name == "type_accuracy":
            print(f"- 16-class type accuracy: {metrics:.4f}")
        else:
            print(
                f"- {name}: accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()

    clf = train_model(data_dir, results_dir) if args.train else load_model(results_dir)
    df_test = load_test_data(data_dir)
    evaluate_model(clf, df_test)

    print("\nSample prediction")
    print(f"Text: {args.sample}")
    print(f"Predicted MBTI: {clf.predict(args.sample)}")


if __name__ == "__main__":
    main()
