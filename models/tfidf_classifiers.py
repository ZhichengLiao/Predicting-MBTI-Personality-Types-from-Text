"""
TF-IDF Classifiers for MBTI Personality Type Prediction

Implements:
1. TF-IDF + Logistic Regression
2. TF-IDF + Multinomial Naive Bayes

Author: Kaiyu Liu
"""

import re
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MBTI_TYPES = [
    'INFP', 'INFJ', 'INTP', 'INTJ',
    'ISFP', 'ISFJ', 'ISTP', 'ISTJ',
    'ENFP', 'ENFJ', 'ENTP', 'ENTJ',
    'ESFP', 'ESFJ', 'ESTP', 'ESTJ',
]


def preprocess_posts(posts: str) -> str:
    """
    Basic preprocessing for raw posts text.

    Splits on the ||| delimiter used in the dataset, lowercases, removes URLs
    and non-alphabetic characters. Use this when clean_posts is unavailable.

    Args:
        posts: Raw posts string (may be ||| separated).

    Returns:
        Single cleaned text string.
    """
    text = posts.replace('|||', ' ')
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TFIDFLogisticRegression:
    """
    MBTI classifier using TF-IDF features and Logistic Regression.

    TF-IDF (Term Frequency - Inverse Document Frequency) weights each term by
    how often it appears in a document relative to the whole corpus.
    Logistic Regression is then trained on these sparse feature vectors for
    16-class MBTI classification.
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = 'balanced',
    ):
        """
        Args:
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: N-gram range. (1,2) includes unigrams and bigrams.
            min_df: Minimum document frequency to include a term.
            C: Regularization strength for Logistic Regression (inverse).
            max_iter: Max solver iterations.
            class_weight: 'balanced' handles class imbalance automatically.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,  # Apply log(1 + tf) to dampen high-frequency terms
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver='lbfgs',
            multi_class='multinomial',
            n_jobs=-1,
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> 'TFIDFLogisticRegression':
        """
        Fit the TF-IDF vectorizer and train Logistic Regression.

        Args:
            texts: List of preprocessed text strings.
            labels: List of MBTI type labels.

        Returns:
            self
        """
        logger.info("Fitting TF-IDF + Logistic Regression...")
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"  Feature matrix shape: {X.shape}")

        y = self.label_encoder.fit_transform(labels)
        logger.info("  Training Logistic Regression...")
        self.classifier.fit(X, y)
        self.is_fitted = True
        logger.info("  Done.")
        return self

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict MBTI type(s).

        Args:
            texts: Single text or list of texts.

        Returns:
            Predicted MBTI type(s).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        predictions = list(self.label_encoder.inverse_transform(y_pred))
        return predictions[0] if single else predictions

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get probability estimates for all 16 MBTI types.

        Args:
            texts: Single text or list of texts.

        Returns:
            Array of shape (n_texts, 16) with class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def get_type_labels(self) -> List[str]:
        """Return MBTI type labels in the order used by predict_proba."""
        return list(self.label_encoder.classes_) if self.is_fitted else []


class TFIDFNaiveBayes:
    """
    MBTI classifier using TF-IDF features and Multinomial Naive Bayes.

    Naive Bayes is a strong, fast baseline for text classification.
    MultinomialNB works with non-negative feature values, which TF-IDF satisfies
    (log(1+tf) >= 0).
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        alpha: float = 0.1,
    ):
        """
        Args:
            max_features: Maximum vocabulary size.
            ngram_range: N-gram range.
            min_df: Minimum document frequency.
            alpha: Laplace smoothing parameter for Naive Bayes.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.classifier = MultinomialNB(alpha=alpha)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> 'TFIDFNaiveBayes':
        """
        Fit the TF-IDF vectorizer and train Naive Bayes.

        Args:
            texts: List of preprocessed text strings.
            labels: List of MBTI type labels.

        Returns:
            self
        """
        logger.info("Fitting TF-IDF + Naive Bayes...")
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"  Feature matrix shape: {X.shape}")

        y = self.label_encoder.fit_transform(labels)
        self.classifier.fit(X, y)
        self.is_fitted = True
        logger.info("  Done.")
        return self

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict MBTI type(s).

        Args:
            texts: Single text or list of texts.

        Returns:
            Predicted MBTI type(s).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        predictions = list(self.label_encoder.inverse_transform(y_pred))
        return predictions[0] if single else predictions

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get probability estimates for all 16 MBTI types.

        Args:
            texts: Single text or list of texts.

        Returns:
            Array of shape (n_texts, 16) with class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def get_type_labels(self) -> List[str]:
        """Return MBTI type labels in the order used by predict_proba."""
        return list(self.label_encoder.classes_) if self.is_fitted else []


class TFIDFPipeline:
    """
    Unified pipeline for TF-IDF-based MBTI classification.

    Trains both Logistic Regression and Naive Bayes classifiers and provides
    a shared interface for prediction, evaluation, and serialization.
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        lr_C: float = 1.0,
        nb_alpha: float = 0.1,
    ):
        """
        Args:
            max_features: Maximum TF-IDF vocabulary size.
            ngram_range: N-gram range shared by both classifiers.
            lr_C: Logistic Regression regularization strength.
            nb_alpha: Naive Bayes Laplace smoothing parameter.
        """
        self.lr_clf = TFIDFLogisticRegression(
            max_features=max_features, ngram_range=ngram_range, C=lr_C
        )
        self.nb_clf = TFIDFNaiveBayes(
            max_features=max_features, ngram_range=ngram_range, alpha=nb_alpha
        )
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> 'TFIDFPipeline':
        """
        Train both classifiers.

        Args:
            texts: List of preprocessed text strings.
            labels: List of MBTI type labels.

        Returns:
            self
        """
        logger.info("=" * 60)
        logger.info("Training TF-IDF Pipeline")
        logger.info("=" * 60)
        logger.info("\n[1/2] Logistic Regression")
        self.lr_clf.fit(texts, labels)
        logger.info("\n[2/2] Naive Bayes")
        self.nb_clf.fit(texts, labels)
        self.is_fitted = True
        logger.info("\nTF-IDF Pipeline training complete!")
        return self

    def predict(
        self, texts: Union[str, List[str]], method: str = 'lr'
    ) -> Union[str, List[str]]:
        """
        Predict MBTI type(s).

        Args:
            texts: Single text or list of texts.
            method: 'lr' for Logistic Regression, 'nb' for Naive Bayes.

        Returns:
            Predicted MBTI type(s).
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        if method == 'lr':
            return self.lr_clf.predict(texts)
        elif method == 'nb':
            return self.nb_clf.predict(texts)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'lr' or 'nb'.")

    def predict_proba(
        self, texts: Union[str, List[str]], method: str = 'lr'
    ) -> np.ndarray:
        """
        Get probability estimates.

        Args:
            texts: Single text or list of texts.
            method: 'lr' or 'nb'.

        Returns:
            Array with class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        if method == 'lr':
            return self.lr_clf.predict_proba(texts)
        elif method == 'nb':
            return self.nb_clf.predict_proba(texts)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'lr' or 'nb'.")

    def get_type_labels(self, method: str = 'lr') -> List[str]:
        """Return ordered MBTI type labels for the given method's predict_proba."""
        if method == 'lr':
            return self.lr_clf.get_type_labels()
        return self.nb_clf.get_type_labels()

    def evaluate(
        self, texts: List[str], labels: List[str], method: str = 'lr'
    ) -> dict:
        """
        Evaluate a classifier on a labeled test set.

        Args:
            texts: List of text strings.
            labels: Ground-truth MBTI type labels.
            method: 'lr' or 'nb'.

        Returns:
            dict with 'accuracy' (float) and 'report' (str).
        """
        predictions = self.predict(texts, method=method)
        acc = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, zero_division=0)
        logger.info(f"[{method.upper()}] Accuracy: {acc:.4f}")
        logger.info(f"\n{report}")
        return {'accuracy': acc, 'report': report}

    def save(self, path: Union[str, Path]) -> None:
        """
        Save both trained classifiers to disk.

        Args:
            path: Directory to save model files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'tfidf_lr.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.lr_clf.vectorizer,
                'classifier': self.lr_clf.classifier,
                'label_encoder': self.lr_clf.label_encoder,
            }, f)
        with open(path / 'tfidf_nb.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.nb_clf.vectorizer,
                'classifier': self.nb_clf.classifier,
                'label_encoder': self.nb_clf.label_encoder,
            }, f)
        logger.info(f"TF-IDF pipeline saved to {path}")

    def load(self, path: Union[str, Path]) -> 'TFIDFPipeline':
        """
        Load trained classifiers from disk.

        Args:
            path: Directory containing saved model files.

        Returns:
            self
        """
        path = Path(path)
        with open(path / 'tfidf_lr.pkl', 'rb') as f:
            lr_data = pickle.load(f)
        self.lr_clf.vectorizer = lr_data['vectorizer']
        self.lr_clf.classifier = lr_data['classifier']
        self.lr_clf.label_encoder = lr_data['label_encoder']
        self.lr_clf.is_fitted = True

        with open(path / 'tfidf_nb.pkl', 'rb') as f:
            nb_data = pickle.load(f)
        self.nb_clf.vectorizer = nb_data['vectorizer']
        self.nb_clf.classifier = nb_data['classifier']
        self.nb_clf.label_encoder = nb_data['label_encoder']
        self.nb_clf.is_fitted = True

        self.is_fitted = True
        logger.info(f"TF-IDF pipeline loaded from {path}")
        return self


# =============================================================================
# Main (Example Usage)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TF-IDF Classifiers - Example Usage")
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

    pipeline = TFIDFPipeline()
    pipeline.fit(train_texts, train_labels)

    print("\n--- Logistic Regression ---")
    results_lr = pipeline.evaluate(test_texts, test_labels, method='lr')

    print("\n--- Naive Bayes ---")
    results_nb = pipeline.evaluate(test_texts, test_labels, method='nb')

    # Example single-text prediction
    sample = "I love deep philosophical discussions and spending time alone with my thoughts."
    print(f"\nSample text: '{sample}'")
    print(f"LR prediction:  {pipeline.predict(sample, method='lr')}")
    print(f"NB prediction:  {pipeline.predict(sample, method='nb')}")

    proba = pipeline.predict_proba(sample, method='lr')
    labels = pipeline.get_type_labels(method='lr')
    top3 = np.argsort(proba[0])[-3:][::-1]
    print("\nTop-3 LR predictions:")
    for i in top3:
        print(f"  {labels[i]}: {proba[0][i]:.4f}")
