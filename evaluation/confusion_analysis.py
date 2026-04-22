"""
Dichotomy Confusion Analysis

For each misclassified sample, records which MBTI dimensions were flipped.
Example: true=INFP, pred=ENFP → I/E dimension flipped.

Produces:
  - Per-dimension flip frequency counts
  - Distribution of how many dimensions flip per error
  - Bar chart saved to results/

Author: Yuyang Zeng
"""

from collections import Counter
from pathlib import Path
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import sys

DIM_NAMES = ['E/I', 'S/N', 'T/F', 'J/P']


def analyze_flips(true_labels: List[str], pred_labels: List[str]) -> Dict:
    """
    For every misclassified sample, identify which MBTI dimensions flipped.

    Args:
        true_labels: Ground-truth MBTI types (e.g. ['INFP', 'ESTJ', ...])
        pred_labels: Predicted MBTI types.

    Returns:
        dict with:
          total_samples, total_errors, error_rate,
          flip_counts (per dimension),
          multi_flip_distribution (errors with 1/2/3/4 dims flipped),
          error_details (list of per-error dicts)
    """
    flip_counts = Counter()
    multi_flip  = Counter()
    error_details = []

    for true, pred in zip(true_labels, pred_labels):
        if true.upper() == pred.upper():
            continue
        true, pred = true.upper(), pred.upper()
        flipped = [
            DIM_NAMES[d] for d in range(4) if true[d] != pred[d]
        ]
        for dim in flipped:
            flip_counts[dim] += 1
        multi_flip[len(flipped)] += 1
        error_details.append({
            'true':    true,
            'pred':    pred,
            'flipped': flipped,
        })

    n_errors = len(error_details)
    total    = len(true_labels)
    return {
        'total_samples':          total,
        'total_errors':           n_errors,
        'error_rate':             n_errors / total if total else 0.0,
        'flip_counts':            dict(flip_counts),
        'multi_flip_distribution': dict(multi_flip),
        'error_details':          error_details,
    }


def print_report(report: Dict) -> None:
    print(f"\n=== Dichotomy Confusion Analysis ===")
    print(f"Errors : {report['total_errors']} / {report['total_samples']} "
          f"({report['error_rate']:.1%})\n")

    print("Dimension flip frequency (% of errors that involve each axis):")
    for dim in DIM_NAMES:
        count = report['flip_counts'].get(dim, 0)
        pct   = count / report['total_errors'] if report['total_errors'] else 0.0
        print(f"  {dim} : {count:4d}  ({pct:.1%})")

    print("\nNumber of dimensions flipped per error:")
    for k in sorted(report['multi_flip_distribution']):
        print(f"  {k} dim(s) flipped : {report['multi_flip_distribution'][k]:4d} errors")


def plot_flip_counts(report: Dict, save_path: str = None) -> None:
    counts = [report['flip_counts'].get(d, 0) for d in DIM_NAMES]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-dimension flip bar chart
    bars = axes[0].bar(DIM_NAMES, counts, color=colors)
    axes[0].set_title('Per-Dichotomy Flip Frequency')
    axes[0].set_xlabel('MBTI Dimension')
    axes[0].set_ylabel('# errors involving this dimension')
    for bar, count in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count), ha='center', va='bottom', fontsize=9,
        )

    # Right: multi-flip distribution
    multi = report['multi_flip_distribution']
    xs = sorted(multi.keys())
    ys = [multi[x] for x in xs]
    axes[1].bar([str(x) for x in xs], ys, color='#8172B2')
    axes[1].set_title('Dimensions Flipped per Error')
    axes[1].set_xlabel('Number of dimensions flipped')
    axes[1].set_ylabel('# errors')
    for i, (x, y) in enumerate(zip(xs, ys)):
        axes[1].text(i, y + max(ys) * 0.01, str(y),
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()


# ----------------------------------------------------------------------
# Main — runs dichotomy analysis using trained DichotomyClassifiers
# ----------------------------------------------------------------------

if __name__ == '__main__':
    proj_root = Path(__file__).parent.parent
    sys.path.insert(0, str(proj_root))
    from models.dichotomy_classifiers import DichotomyClassifiers

    data_dir    = proj_root / 'data' / 'processed'
    results_dir = proj_root / 'results'

    df_test = pd.read_csv(data_dir / 'test.csv')
    text_col = 'clean_posts' if 'clean_posts' in df_test.columns else 'posts'

    print("Loading trained model...")
    model_path = results_dir / 'dichotomy_classifiers.pkl'
    if not model_path.exists():
        print("No saved model found — training from scratch...")
        df_train = pd.read_csv(data_dir / 'train.csv.gz')
        clf = DichotomyClassifiers()
        clf.fit_from_df(df_train, text_col=text_col)
        clf.save(results_dir)
    else:
        clf = DichotomyClassifiers().load(results_dir)

    print("Generating predictions...")
    texts = df_test[text_col].fillna('').tolist()
    preds = clf.predict(texts)
    true  = df_test['type'].tolist()

    report = analyze_flips(true, preds)
    print_report(report)

    chart_path = str(results_dir / 'dichotomy_flip_counts.png')
    plot_flip_counts(report, save_path=chart_path)
