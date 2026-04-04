"""
BM25 Retrieval for MBTI Personality Type Prediction

Uses BM25 to retrieve similar training documents for a query,
then classifies by majority vote over the top-k retrieved neighbors.

Author: Kaiyu Liu
"""

import pickle
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MBTI_TYPES = [
    'INFP', 'INFJ', 'INTP', 'INTJ',
    'ISFP', 'ISFJ', 'ISTP', 'ISTJ',
    'ENFP', 'ENFJ', 'ENTP', 'ENTJ',
    'ESFP', 'ESFJ', 'ESTP', 'ESTJ',
]


def tokenize(text: str) -> List[str]:
    """
    Whitespace tokenizer for BM25.

    BM25Okapi expects a list of tokens per document. The clean_posts column
    in the processed dataset is already lowercased and stripped of noise, so
    simple whitespace splitting is sufficient.

    Args:
        text: Input text string.

    Returns:
        List of tokens.
    """
    return text.lower().split()


class BM25Retriever:
    """
    BM25-based k-NN classifier for MBTI type prediction.

    Each training sample is indexed as a BM25 document. To classify a new text
    (query), BM25 scores are computed against all training documents. The top-k
    documents are retrieved and their MBTI labels are majority-voted to produce
    the final prediction.

    BM25 (Okapi BM25) scoring formula:
        score(D, Q) = sum_t IDF(t) * tf(t,D) * (k1+1)
                                     / (tf(t,D) + k1*(1 - b + b*|D|/avgdl))

        where:
          IDF(t)  = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
          k1      = term saturation parameter (default 1.5)
          b       = length normalization parameter (default 0.75)
          |D|     = document length, avgdl = average document length
    """

    def __init__(self, k: int = 10, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k: Number of nearest neighbors to retrieve for majority voting.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
        """
        self.k = k
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.train_labels: Optional[List[str]] = None
        self.train_texts: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> 'BM25Retriever':
        """
        Build the BM25 index from training documents.

        Args:
            texts: List of preprocessed text strings (one per training sample).
            labels: Corresponding MBTI type labels.

        Returns:
            self
        """
        logger.info(f"Building BM25 index over {len(texts)} documents...")
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self.train_labels = list(labels)
        self.train_texts = list(texts)
        self.is_fitted = True
        logger.info("BM25 index built.")
        return self

    def _predict_one(self, text: str) -> str:
        """Predict a single text by majority vote over top-k BM25 results."""
        tokens = tokenize(text)
        scores = self.bm25.get_scores(tokens)
        top_k_indices = np.argsort(scores)[-self.k:][::-1]
        top_k_labels = [self.train_labels[i] for i in top_k_indices]
        # Majority vote; ties broken naturally by Counter.most_common
        return Counter(top_k_labels).most_common(1)[0][0]

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict MBTI type(s) via BM25 retrieval + majority vote.

        Args:
            texts: Single text or list of texts.

        Returns:
            Predicted MBTI type(s).
        """
        if not self.is_fitted:
            raise ValueError("Retriever not fitted. Call fit() first.")
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        predictions = [self._predict_one(t) for t in texts]
        return predictions[0] if single else predictions

    def retrieve(
        self, text: str, top_k: Optional[int] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Retrieve the top-k most similar training documents for a query.

        Useful for inspecting what evidence drives a prediction.

        Args:
            text: Query text.
            top_k: Number of results to return (defaults to self.k).

        Returns:
            List of (mbti_label, text_snippet, bm25_score) tuples,
            sorted by descending score.
        """
        if not self.is_fitted:
            raise ValueError("Retriever not fitted. Call fit() first.")
        k = top_k if top_k is not None else self.k
        tokens = tokenize(text)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-k:][::-1]
        return [
            (self.train_labels[i], self.train_texts[i][:120] + '...', float(scores[i]))
            for i in top_indices
        ]

    def evaluate(self, texts: List[str], labels: List[str]) -> dict:
        """
        Evaluate the BM25 classifier on a labeled test set.

        Args:
            texts: List of text strings.
            labels: Ground-truth MBTI type labels.

        Returns:
            dict with 'accuracy' (float) and 'report' (str).
        """
        logger.info(f"Evaluating BM25 on {len(texts)} samples (k={self.k})...")
        predictions = self.predict(texts)
        acc = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, zero_division=0)
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"\n{report}")
        return {'accuracy': acc, 'report': report}

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted retriever to disk.

        Args:
            path: Directory to save the model file.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'bm25_retriever.pkl', 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'train_labels': self.train_labels,
                'train_texts': self.train_texts,
                'k': self.k,
                'k1': self.k1,
                'b': self.b,
            }, f)
        logger.info(f"BM25 retriever saved to {path}")

    def load(self, path: Union[str, Path]) -> 'BM25Retriever':
        """
        Load a fitted retriever from disk.

        Args:
            path: Directory containing the saved model file.

        Returns:
            self
        """
        path = Path(path)
        with open(path / 'bm25_retriever.pkl', 'rb') as f:
            data = pickle.load(f)
        self.bm25 = data['bm25']
        self.train_labels = data['train_labels']
        self.train_texts = data['train_texts']
        self.k = data['k']
        self.k1 = data['k1']
        self.b = data['b']
        self.is_fitted = True
        logger.info(f"BM25 retriever loaded from {path}")
        return self


# =============================================================================
# Main (Example Usage)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("BM25 Retrieval - Example Usage")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'

    if not train_path.exists():
        print(f"train.csv not found at {train_path}. Splitting test.csv for demo.")
        df = pd.read_csv(test_path)
        df_test = df.sample(frac=0.2, random_state=42)
        df_train = df.drop(df_test.index)
    else:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

    # Use pre-cleaned text if available
    text_col = 'clean_posts' if 'clean_posts' in df_train.columns else 'posts'
    train_texts = df_train[text_col].fillna('').tolist()
    train_labels = df_train['type'].tolist()
    test_texts = df_test[text_col].fillna('').tolist()
    test_labels = df_test['type'].tolist()

    retriever = BM25Retriever(k=10)
    retriever.fit(train_texts, train_labels)
    retriever.evaluate(test_texts, test_labels)

    # Inspect retrieved evidence for one test sample
    print("\n--- Top-5 Retrieved Docs for First Test Sample ---")
    results = retriever.retrieve(test_texts[0], top_k=5)
    for label, snippet, score in results:
        print(f"  [{label}] score={score:.2f}  {snippet}")
