"""
Embedding Pipeline for MBTI Personality Type Prediction

This module implements dense retrieval approaches for MBTI classification:
1. Sentence embeddings using all-MiniLM-L6-v2
2. Per-type centroid classification via cosine similarity
3. Logistic Regression on embedding features

Author: Zhicheng Liao (zl139@illinois.edu)
Course: CS410 - Text Information Systems
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All 16 MBTI types
MBTI_TYPES = [
    'INFP', 'INFJ', 'INTP', 'INTJ',
    'ISFP', 'ISFJ', 'ISTP', 'ISTJ',
    'ENFP', 'ENFJ', 'ENTP', 'ENTJ',
    'ESFP', 'ESFJ', 'ESTP', 'ESTJ'
]


class EmbeddingModel:
    """
    Wrapper for sentence-transformers model.

    Uses all-MiniLM-L6-v2 to encode text into 384-dimensional dense vectors.
    This model is optimized for semantic similarity tasks.

    Reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformer model to use.
                        Default is 'all-MiniLM-L6-v2' (384 dimensions).
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None

    def load(self) -> 'EmbeddingModel':
        """Load the sentence-transformer model."""
        logger.info(f"Loading sentence-transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        # Get embedding dimension from a test encode
        test_embedding = self.model.encode(["test"], show_progress_bar=False)
        self.embedding_dim = test_embedding.shape[1]
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        return self

    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 32,
               show_progress: bool = True,
               normalize: bool = True) -> np.ndarray:
        """
        Encode texts into dense vector embeddings.

        Args:
            texts: Single text or list of texts to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity).

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if self.model is None:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
        return embeddings


class CentroidClassifier:
    """
    MBTI classifier using per-type centroid embeddings and cosine similarity.

    This approach treats each MBTI type as a "document collection" and computes
    a centroid (mean) embedding for each type. Classification is done by finding
    the type whose centroid is most similar to the query embedding.

    This directly applies the Dense Passage Retrieval paradigm to personality
    prediction (Karpukhin et al., 2020).
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize the centroid classifier.

        Args:
            embedding_model: Pre-initialized EmbeddingModel instance.
                           If None, a new one will be created.
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.centroids: Optional[Dict[str, np.ndarray]] = None
        self.centroid_matrix: Optional[np.ndarray] = None
        self.type_order: Optional[List[str]] = None

    def fit(self, texts: List[str], labels: List[str], embeddings: Optional[np.ndarray] = None) -> 'CentroidClassifier':
        """
        Compute centroid embeddings for each MBTI type.

        Args:
            texts: List of text samples (user posts).
            labels: List of corresponding MBTI type labels.
            embeddings: Optional pre-computed embeddings. If None, texts will be encoded.

        Returns:
            self
        """
        logger.info("Computing centroid embeddings for each MBTI type...")

        # Ensure model is loaded
        if self.embedding_model.model is None:
            self.embedding_model.load()

        if embeddings is None:
            # Encode all texts
            logger.info(f"Encoding {len(texts)} texts...")
            embeddings = self.embedding_model.encode(texts, show_progress=True)
        else:
            logger.info(f"Using {len(embeddings)} pre-computed embeddings...")

        # Group embeddings by MBTI type and compute centroids
        self.centroids = {}
        df = pd.DataFrame({'embedding': list(embeddings), 'label': labels})

        for mbti_type in MBTI_TYPES:
            type_embeddings = df[df['label'] == mbti_type]['embedding'].tolist()
            if len(type_embeddings) > 0:
                # Compute mean centroid
                centroid = np.mean(np.stack(type_embeddings), axis=0)
                # L2 normalize the centroid
                centroid = centroid / np.linalg.norm(centroid)
                self.centroids[mbti_type] = centroid
                logger.info(f"  {mbti_type}: {len(type_embeddings)} samples")
            else:
                logger.warning(f"  {mbti_type}: No samples found!")

        # Create centroid matrix for efficient batch prediction
        self.type_order = list(self.centroids.keys())
        self.centroid_matrix = np.stack([self.centroids[t] for t in self.type_order])

        logger.info(f"Centroids computed for {len(self.centroids)} MBTI types.")
        return self

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict MBTI type(s) using cosine similarity to centroids.

        Args:
            texts: Single text or list of texts to classify.

        Returns:
            Predicted MBTI type(s).
        """
        if self.centroid_matrix is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode query texts
        query_embeddings = self.embedding_model.encode(texts, show_progress=False)

        # Compute cosine similarities with all centroids
        # Shape: (n_queries, n_types)
        similarities = cosine_similarity(query_embeddings, self.centroid_matrix)

        # Get the type with highest similarity for each query
        best_indices = np.argmax(similarities, axis=1)
        predictions = [self.type_order[i] for i in best_indices]

        return predictions[0] if single_input else predictions

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get similarity scores (pseudo-probabilities) for all MBTI types.

        Args:
            texts: Single text or list of texts.

        Returns:
            numpy array of shape (n_texts, n_types) with similarity scores.
        """
        if self.centroid_matrix is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        if isinstance(texts, str):
            texts = [texts]

        query_embeddings = self.embedding_model.encode(texts, show_progress=False)
        similarities = cosine_similarity(query_embeddings, self.centroid_matrix)

        # Apply softmax for probability-like scores
        exp_sim = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
        proba = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        return proba

    def get_type_labels(self) -> List[str]:
        """Get the ordered list of MBTI type labels."""
        return self.type_order if self.type_order else []


class EmbeddingLogisticRegression:
    """
    MBTI classifier using embedding features with Logistic Regression.

    This approach uses sentence embeddings as features and trains a
    Logistic Regression classifier for the 16-class classification task.
    """

    def __init__(self,
                 embedding_model: Optional[EmbeddingModel] = None,
                 C: float = 1.0,
                 max_iter: int = 1000,
                 class_weight: Optional[str] = 'balanced'):
        """
        Initialize the embedding + LR classifier.

        Args:
            embedding_model: Pre-initialized EmbeddingModel instance.
            C: Regularization strength (inverse).
            max_iter: Maximum iterations for LR solver.
            class_weight: Class weight strategy ('balanced' recommended for imbalanced data).
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver='lbfgs',
            multi_class='multinomial',
            n_jobs=-1,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str], embeddings: Optional[np.ndarray] = None) -> 'EmbeddingLogisticRegression':
        """
        Train the Logistic Regression classifier on embedding features.

        Args:
            texts: List of text samples.
            labels: List of corresponding MBTI type labels.
            embeddings: Optional pre-computed embeddings. If None, texts will be encoded.

        Returns:
            self
        """
        logger.info("Training Embedding + Logistic Regression classifier...")

        # Ensure model is loaded
        if self.embedding_model.model is None:
            self.embedding_model.load()

        if embeddings is None:
            # Encode all texts
            logger.info(f"Encoding {len(texts)} texts...")
            X = self.embedding_model.encode(texts, show_progress=True)
        else:
            logger.info(f"Using {len(embeddings)} pre-computed embeddings...")
            X = embeddings

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Train classifier
        logger.info("Training Logistic Regression...")
        self.classifier.fit(X, y)
        self.is_fitted = True

        logger.info("Training complete.")
        return self

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict MBTI type(s) using the trained classifier.

        Args:
            texts: Single text or list of texts to classify.

        Returns:
            Predicted MBTI type(s).
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode and predict
        X = self.embedding_model.encode(texts, show_progress=False)
        y_pred = self.classifier.predict(X)
        predictions = self.label_encoder.inverse_transform(y_pred)

        return predictions[0] if single_input else list(predictions)

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get probability estimates for all MBTI types.

        Args:
            texts: Single text or list of texts.

        Returns:
            numpy array of shape (n_texts, n_types) with probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        if isinstance(texts, str):
            texts = [texts]

        X = self.embedding_model.encode(texts, show_progress=False)
        return self.classifier.predict_proba(X)

    def get_type_labels(self) -> List[str]:
        """Get the ordered list of MBTI type labels."""
        return list(self.label_encoder.classes_) if self.is_fitted else []


class EmbeddingPipeline:
    """
    Unified pipeline for embedding-based MBTI personality prediction.

    Combines both approaches:
    1. Centroid-based cosine similarity classification
    2. Logistic Regression on embedding features

    Provides a unified interface for training, prediction, and evaluation.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding pipeline.

        Args:
            model_name: Name of the sentence-transformer model.
        """
        self.model_name = model_name
        self.embedding_model = EmbeddingModel(model_name)
        self.centroid_clf = CentroidClassifier(self.embedding_model)
        self.lr_clf = EmbeddingLogisticRegression(self.embedding_model)
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> 'EmbeddingPipeline':
        """
        Train both classifiers on the provided data.

        Args:
            texts: List of text samples (user posts).
            labels: List of corresponding MBTI type labels.

        Returns:
            self
        """
        logger.info("=" * 60)
        logger.info("Training Embedding Pipeline")
        loCompute embeddings once
        logger.info(f"Encoding {len(texts)} texts for shared use...")
        embeddings = self.embedding_model.encode(texts, show_progress=True)

        # Train centroid classifier
        logger.info("\n[1/2] Training Centroid Classifier...")
        self.centroid_clf.fit(texts, labels, embeddings=embeddings)

        # Train LR classifier (reuses cached embeddings conceptually)
        logger.info("\n[2/2] Training Logistic Regression Classifier...")
        self.lr_clf.fit(texts, labels, embeddings=embeddingg Centroid Classifier...")
        self.centroid_clf.fit(texts, labels)

        # Train LR classifier (reuses cached embeddings conceptually)
        logger.info("\n[2/2] Training Logistic Regression Classifier...")
        self.lr_clf.fit(texts, labels)

        self.is_fitted = True
        logger.info("\n" + "=" * 60)
        logger.info("Embedding Pipeline training complete!")
        logger.info("=" * 60)

        return self

    def predict(self,
                texts: Union[str, List[str]],
                method: str = 'lr') -> Union[str, List[str]]:
        """
        Predict MBTI type(s) using specified method.

        Args:
            texts: Single text or list of texts to classify.
            method: Classification method - 'centroid' or 'lr' (default).

        Returns:
            Predicted MBTI type(s).
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if method == 'centroid':
            return self.centroid_clf.predict(texts)
        elif method == 'lr':
            return self.lr_clf.predict(texts)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'centroid' or 'lr'.")

    def predict_proba(self,
                      texts: Union[str, List[str]],
                      method: str = 'lr') -> np.ndarray:
        """
        Get probability estimates for all MBTI types.

        Args:
            texts: Single text or list of texts.
            method: Classification method - 'centroid' or 'lr' (default).

        Returns:
            numpy array with probability/similarity scores.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if method == 'centroid':
            return self.centroid_clf.predict_proba(texts)
        elif method == 'lr':
            return self.lr_clf.predict_proba(texts)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'centroid' or 'lr'.")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get raw embedding vectors for texts.

        Useful for downstream tasks or analysis.

        Args:
            texts: Single text or list of texts.

        Returns:
            numpy array of embeddings.
        """
        return self.embedding_model.encode(texts)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained pipeline to disk.

        Args:
            path: Directory path to save the pipeline.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save centroid data
        centroid_data = {
            'centroids': self.centroid_clf.centroids,
            'centroid_matrix': self.centroid_clf.centroid_matrix,
            'type_order': self.centroid_clf.type_order
        }
        with open(path / 'centroid_clf.pkl', 'wb') as f:
            pickle.dump(centroid_data, f)

        # Save LR classifier
        lr_data = {
            'classifier': self.lr_clf.classifier,
            'label_encoder': self.lr_clf.label_encoder
        }
        with open(path / 'lr_clf.pkl', 'wb') as f:
            pickle.dump(lr_data, f)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_model.embedding_dim
        }
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Pipeline saved to {path}")

    def load(self, path: Union[str, Path]) -> 'EmbeddingPipeline':
        """
        Load a trained pipeline from disk.

        Args:
            path: Directory path containing saved pipeline.

        Returns:
            self
        """
        path = Path(path)

        # Load metadata
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        self.model_name = metadata['model_name']

        # Reload embedding model
        self.embedding_model = EmbeddingModel(self.model_name)
        self.embedding_model.load()
        self.embedding_model.embedding_dim = metadata['embedding_dim']

        # Load centroid classifier
        with open(path / 'centroid_clf.pkl', 'rb') as f:
            centroid_data = pickle.load(f)
        self.centroid_clf = CentroidClassifier(self.embedding_model)
        self.centroid_clf.centroids = centroid_data['centroids']
        self.centroid_clf.centroid_matrix = centroid_data['centroid_matrix']
        self.centroid_clf.type_order = centroid_data['type_order']

        # Load LR classifier
        with open(path / 'lr_clf.pkl', 'rb') as f:
            lr_data = pickle.load(f)
        self.lr_clf = EmbeddingLogisticRegression(self.embedding_model)
        self.lr_clf.classifier = lr_data['classifier']
        self.lr_clf.label_encoder = lr_data['label_encoder']
        self.lr_clf.is_fitted = True

        self.is_fitted = True
        logger.info(f"Pipeline loaded from {path}")

        return self


# =============================================================================
# Utility Functions
# =============================================================================

def compute_embeddings_for_dataset(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute embeddings for an entire dataset.

    Utility function for pre-computing embeddings to avoid redundant computation.

    Args:
        texts: List of texts to encode.
        model_name: Sentence-transformer model name.
        batch_size: Batch size for encoding.
        output_path: Optional path to save embeddings as .npy file.

    Returns:
        numpy array of embeddings.
    """
    model = EmbeddingModel(model_name)
    model.load()

    embeddings = model.encode(texts, batch_size=batch_size, show_progress=True)

    if output_path:
        np.save(output_path, embeddings)
        logger.info(f"Embeddings saved to {output_path}")

    return embeddings


# =============================================================================
# Main (Example Usage)
# =============================================================================

if __name__ == '__main__':
    # Example usage demonstration
    print("=" * 60)
    print("MBTI Embedding Pipeline - Example Usage")
    print("=" * 60)

    # Sample data (in practice, load from data/processed/train.csv)
    sample_texts = [
        "I love spending time alone reading books and thinking deeply about life.",
        "Let's go to the party! I can't wait to meet new people and have fun!",
        "I prefer logical analysis and systematic approaches to solve problems.",
        "I care deeply about others' feelings and always try to help people.",
    ]
    sample_labels = ['INFP', 'ENFP', 'INTP', 'ENFJ']

    # Initialize and train pipeline
    pipeline = EmbeddingPipeline()
    pipeline.fit(sample_texts, sample_labels)

    # Test prediction
    test_text = "I enjoy quiet evenings with a good book rather than loud parties."

    print(f"\nTest text: '{test_text}'")
    print(f"Centroid prediction: {pipeline.predict(test_text, method='centroid')}")
    print(f"LR prediction: {pipeline.predict(test_text, method='lr')}")

    # Get probabilities
    proba = pipeline.predict_proba(test_text, method='lr')
    labels = pipeline.lr_clf.get_type_labels()
    print(f"\nTop 3 predictions:")
    for idx in np.argsort(proba[0])[-3:][::-1]:
        print(f"  {labels[idx]}: {proba[0][idx]:.4f}")
