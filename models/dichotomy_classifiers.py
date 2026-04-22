"""
Per-Dichotomy Binary Classifiers for MBTI Prediction

Four independent TF-IDF + Logistic Regression binary classifiers,
one per MBTI dimension (E/I, S/N, T/F, J/P).
Predictions are concatenated into a full 4-letter MBTI type.

Author: Yuyang Zeng
"""

import pickle
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column names in the processed CSVs, in MBTI letter order
DIM_COLS   = ['IE', 'SN', 'TF', 'JP']
DIM_LABELS = ['E/I', 'S/N', 'T/F', 'J/P']


class DichotomyClassifiers:
    """
    Four independent TF-IDF + Logistic Regression binary classifiers,
    one per MBTI dimension.

    Each classifier is trained on its own TF-IDF vectorizer so that
    vocabulary selection is independent per dimension.
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range=(1, 2),
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.C            = C
        self.max_iter     = max_iter

        self.vectorizers = [
            TfidfVectorizer(
                max_features=max_features, ngram_range=ngram_range,
                min_df=2, sublinear_tf=True,
            )
            for _ in range(4)
        ]
        self.classifiers = [
            LogisticRegression(
                C=C, max_iter=max_iter, class_weight='balanced',
                solver='lbfgs', n_jobs=-1, random_state=42,
            )
            for _ in range(4)
        ]
        self.encoders = [LabelEncoder() for _ in range(4)]
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: List[str],
        dim_labels: List[List[str]],
    ) -> 'DichotomyClassifiers':
        """
        Train all four binary classifiers.

        Args:
            texts:      List of pre-cleaned text strings (one per sample).
            dim_labels: List of four label-lists, one per dimension.
                        e.g. [IE_labels, SN_labels, TF_labels, JP_labels]
                        Each inner list has values like ['I','E','I',...].
        """
        for dim in range(4):
            logger.info(f"Training dim {dim} ({DIM_LABELS[dim]})...")
            X = self.vectorizers[dim].fit_transform(texts)
            y = self.encoders[dim].fit_transform(dim_labels[dim])
            self.classifiers[dim].fit(X, y)
            logger.info(
                f"  classes: {self.encoders[dim].classes_}  "
                f"vocab: {len(self.vectorizers[dim].vocabulary_)}"
            )
        self.is_fitted = True
        logger.info("All four dichotomy classifiers trained.")
        return self

    def fit_from_df(self, df: pd.DataFrame, text_col: str = 'clean_posts') -> 'DichotomyClassifiers':
        """Convenience wrapper: train directly from a processed DataFrame."""
        texts = df[text_col].fillna('').tolist()
        dim_labels = [df[col].tolist() for col in DIM_COLS]
        return self.fit(texts, dim_labels)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_dim(self, dim: int, texts: List[str]) -> List[str]:
        X = self.vectorizers[dim].transform(texts)
        y_enc = self.classifiers[dim].predict(X)
        return list(self.encoders[dim].inverse_transform(y_enc))

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict full MBTI type by concatenating four binary predictions.

        Returns a single string if input is a string, else a list.
        """
        if not self.is_fitted:
            raise ValueError("Not fitted. Call fit() or fit_from_df() first.")
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        per_dim = [self._predict_dim(d, texts) for d in range(4)]
        results = [''.join(per_dim[d][i] for d in range(4)) for i in range(len(texts))]
        return results[0] if single else results

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Return (n, 4) array of probabilities for the second class of each
        dimension (class index 1 according to LabelEncoder ordering).
        """
        if not self.is_fitted:
            raise ValueError("Not fitted.")
        if isinstance(texts, str):
            texts = [texts]
        proba = np.zeros((len(texts), 4))
        for dim in range(4):
            X = self.vectorizers[dim].transform(texts)
            proba[:, dim] = self.classifiers[dim].predict_proba(X)[:, 1]
        return proba

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        texts: List[str],
        dim_labels: List[List[str]],
        type_labels: List[str] = None,
    ) -> dict:
        """
        Evaluate per-dimension accuracy + macro F1.
        Optionally evaluates full 16-class type accuracy if type_labels given.

        Returns a dict with per-dimension metrics and (optionally) type_accuracy.
        """
        results = {}
        for dim in range(4):
            y_true_str = dim_labels[dim]
            y_pred_str = self._predict_dim(dim, texts)
            y_true_enc = self.encoders[dim].transform(y_true_str)
            y_pred_enc = self.encoders[dim].transform(y_pred_str)

            acc    = accuracy_score(y_true_enc, y_pred_enc)
            macro_f1 = f1_score(y_true_enc, y_pred_enc, average='macro')
            report = classification_report(
                y_true_str, y_pred_str, zero_division=0
            )
            results[DIM_LABELS[dim]] = {
                'accuracy': acc,
                'macro_f1': macro_f1,
                'report': report,
            }
            logger.info(
                f"[{DIM_LABELS[dim]}] acc={acc:.4f}  macro_f1={macro_f1:.4f}"
            )

        if type_labels is not None:
            preds = self.predict(texts)
            type_acc = accuracy_score(type_labels, preds)
            results['type_accuracy'] = type_acc
            logger.info(f"16-class type accuracy: {type_acc:.4f}")

        return results

    def evaluate_from_df(self, df: pd.DataFrame, text_col: str = 'clean_posts') -> dict:
        """Convenience wrapper: evaluate directly from a processed DataFrame."""
        texts      = df[text_col].fillna('').tolist()
        dim_labels = [df[col].tolist() for col in DIM_COLS]
        type_labels = df['type'].tolist() if 'type' in df.columns else None
        return self.evaluate(texts, dim_labels, type_labels)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'dichotomy_classifiers.pkl', 'wb') as f:
            pickle.dump({
                'vectorizers': self.vectorizers,
                'classifiers': self.classifiers,
                'encoders':    self.encoders,
            }, f)
        logger.info(f"Saved to {path / 'dichotomy_classifiers.pkl'}")

    def load(self, path: Union[str, Path]) -> 'DichotomyClassifiers':
        path = Path(path)
        with open(path / 'dichotomy_classifiers.pkl', 'rb') as f:
            data = pickle.load(f)
        self.vectorizers = data['vectorizers']
        self.classifiers = data['classifiers']
        self.encoders    = data['encoders']
        self.is_fitted   = True
        logger.info(f"Loaded from {path / 'dichotomy_classifiers.pkl'}")
        return self


# ----------------------------------------------------------------------
# Main — example usage
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from pathlib import Path

    proj_root = Path(__file__).parent.parent
    data_dir  = proj_root / 'data' / 'processed'
    results_dir = proj_root / 'results'

    print("Loading data...")
    df_train = pd.read_csv(data_dir / 'train.csv.gz')
    df_test  = pd.read_csv(data_dir / 'test.csv')
    print(f"Train: {len(df_train)}  Test: {len(df_test)}")

    clf = DichotomyClassifiers()
    clf.fit_from_df(df_train)

    print("\n=== Evaluation on test set ===")
    results = clf.evaluate_from_df(df_test)

    print("\n=== Per-dimension classification reports ===")
    for dim_name, metrics in results.items():
        if dim_name == 'type_accuracy':
            print(f"\n16-class type accuracy: {metrics:.4f}")
        else:
            print(f"\n--- {dim_name} ---")
            print(metrics['report'])

    clf.save(results_dir)

    # Quick inference demo
    sample = "I love deep philosophical discussions and spending time alone with my thoughts."
    pred = clf.predict(sample)
    print(f"\nSample text: '{sample}'")
    print(f"Predicted MBTI type: {pred}")
