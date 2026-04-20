"""
Embedding pipeline for MBTI personality type prediction.

This module supports two embedding backends:
1. `sentence-transformers/all-MiniLM-L6-v2` as the baseline.
2. `BAAI/bge-base-en-v1.5` as a controlled longer-context comparison.

It also supports two document preparation modes:
1. `clean_posts`: use the cleaned whole-user text column.
2. `posts_list_pool`: mask MBTI mentions, encode individual posts, then mean-pool
   them into one user embedding.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All 16 MBTI types
MBTI_TYPES = [
    "INFP",
    "INFJ",
    "INTP",
    "INTJ",
    "ISFP",
    "ISFJ",
    "ISTP",
    "ISTJ",
    "ENFP",
    "ENFJ",
    "ENTP",
    "ENTJ",
    "ESFP",
    "ESFJ",
    "ESTP",
    "ESTJ",
]

MBTI_TOKEN_PATTERN = re.compile(
    r"\b(?:infp|infj|intp|intj|isfp|isfj|istp|istj|"
    r"enfp|enfj|entp|entj|esfp|esfj|estp|estj)(?:s)?\b",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")
ALPHABETIC_PATTERN = re.compile(r"[A-Za-z]")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LONG_CONTEXT_MODEL = "BAAI/bge-base-en-v1.5"

MODEL_ALIASES = {
    "minilm": DEFAULT_MODEL,
    "all-minilm": DEFAULT_MODEL,
    "bge-base": LONG_CONTEXT_MODEL,
    "bge-base-en-v1.5": LONG_CONTEXT_MODEL,
}

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    DEFAULT_MODEL: {
        "backend": "sentence_transformers",
        "max_length": 256,
        "batch_size": 64,
    },
    LONG_CONTEXT_MODEL: {
        "backend": "sentence_transformers",
        "max_length": 512,
        "batch_size": 24,
    },
}

TextSample = Union[str, List[str]]


def resolve_model_name(model_name: str) -> str:
    """Resolve a short alias to a full model identifier."""
    return MODEL_ALIASES.get(model_name, model_name)


def resolve_data_path(data_dir: Union[str, Path], split_name: str) -> Path:
    """Resolve train/test split paths, allowing either .csv or .csv.gz."""
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{split_name}.csv"
    gzip_path = data_dir / f"{split_name}.csv.gz"

    if csv_path.exists():
        return csv_path
    if gzip_path.exists():
        return gzip_path

    raise FileNotFoundError(
        f"Could not find {split_name}.csv or {split_name}.csv.gz in {data_dir}"
    )


def load_split_dataframe(data_dir: Union[str, Path], split_name: str) -> pd.DataFrame:
    """Load a train/test split from disk."""
    path = resolve_data_path(data_dir, split_name)
    logger.info("Loading %s split from %s", split_name, path)
    return pd.read_csv(path, compression="infer")


def mask_mbti_mentions(text: str, replacement: str = "MBTI_TYPE") -> str:
    """Mask direct MBTI type mentions to reduce leakage."""
    return MBTI_TOKEN_PATTERN.sub(replacement, text)


def normalize_post_text(text: Any, mask_mbti: bool = True) -> str:
    """Apply lightweight normalization suitable for sentence encoders."""
    if not isinstance(text, str):
        return ""

    normalized = text.replace("|||", " ")
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = normalized.replace("\n", " ").replace("\r", " ")
    normalized = normalized.strip().strip("'").strip('"')

    if mask_mbti:
        normalized = mask_mbti_mentions(normalized)

    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def parse_posts_list(row: pd.Series) -> List[str]:
    """Recover the list of posts for one user."""
    posts_list = row.get("posts_list")
    if isinstance(posts_list, list):
        return posts_list

    if isinstance(posts_list, str) and posts_list.strip():
        try:
            parsed = ast.literal_eval(posts_list)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            logger.debug("Failed to parse posts_list; falling back to posts split.")

    posts = row.get("posts", "")
    if isinstance(posts, str) and posts:
        return posts.split("|||")

    return []


def select_posts(posts: Sequence[str], max_posts: Optional[int]) -> List[str]:
    """Keep the longest informative posts to control runtime."""
    if max_posts is None or len(posts) <= max_posts:
        return list(posts)
    return sorted(posts, key=len, reverse=True)[:max_posts]


def build_documents(
    df: pd.DataFrame,
    input_mode: str = "posts_list_pool",
    mask_mbti: bool = True,
    max_posts: Optional[int] = 16,
    min_post_words: int = 3,
) -> tuple[List[TextSample], Dict[str, Any]]:
    """Build model inputs from a dataframe."""
    documents: List[TextSample] = []

    if input_mode == "clean_posts":
        for _, row in df.iterrows():
            base_text = row.get("clean_posts") or row.get("posts") or ""
            documents.append(normalize_post_text(base_text, mask_mbti=mask_mbti))
        stats = {
            "input_mode": input_mode,
            "num_documents": len(documents),
        }
        return documents, stats

    raw_post_counts: List[int] = []
    kept_post_counts: List[int] = []

    for _, row in df.iterrows():
        raw_posts = parse_posts_list(row)
        raw_post_counts.append(len(raw_posts))

        normalized_posts = []
        for post in raw_posts:
            cleaned = normalize_post_text(post, mask_mbti=mask_mbti)
            if len(cleaned.split()) < min_post_words:
                continue
            if not ALPHABETIC_PATTERN.search(cleaned):
                continue
            normalized_posts.append(cleaned)

        normalized_posts = select_posts(normalized_posts, max_posts=max_posts)

        if not normalized_posts:
            fallback_text = normalize_post_text(
                row.get("clean_posts") or row.get("posts") or "",
                mask_mbti=mask_mbti,
            )
            normalized_posts = [fallback_text] if fallback_text else [""]

        kept_post_counts.append(len(normalized_posts))
        documents.append(normalized_posts)

    stats = {
        "input_mode": input_mode,
        "num_documents": len(documents),
        "avg_raw_posts": round(float(np.mean(raw_post_counts)), 2),
        "avg_kept_posts": round(float(np.mean(kept_post_counts)), 2),
        "max_posts": max_posts,
        "min_post_words": min_post_words,
    }
    return documents, stats


class EmbeddingModel:
    """
    Wrapper over either sentence-transformers or a direct HF encoder model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        self.model_name = resolve_model_name(model_name)
        self.model_spec = MODEL_SPECS.get(
            self.model_name, {"backend": "sentence_transformers"}
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length or self.model_spec.get("max_length")
        self.model = None
        self.tokenizer = None
        self.embedding_dim: Optional[int] = None

    def load(self) -> "EmbeddingModel":
        """Load the configured embedding model."""
        logger.info("Loading embedding model: %s", self.model_name)

        backend = self.model_spec.get("backend", "sentence_transformers")
        if backend == "sentence_transformers":
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_length:
                self.model.max_seq_length = self.max_length
            test_embedding = self.model.encode(
                ["test"], show_progress_bar=False, normalize_embeddings=True
            )
        elif backend == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.model_spec.get("trust_remote_code", False),
            )
            self.model.to(self.device)
            self.model.eval()
            test_embedding = self._encode_text_batch(
                ["test"], batch_size=1, show_progress=False, normalize=True
            )
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")

        self.embedding_dim = int(test_embedding.shape[1])
        logger.info(
            "Model loaded. Backend=%s, device=%s, embedding_dim=%s, max_length=%s",
            backend,
            self.device,
            self.embedding_dim,
            self.max_length,
        )
        return self

    def _encode_text_batch(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        normalize: bool,
    ) -> np.ndarray:
        """Encode a flat list of text strings."""
        if self.model is None:
            self.load()

        if not texts:
            if self.embedding_dim is None:
                return np.empty((0, 0), dtype=np.float32)
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        backend = self.model_spec.get("backend", "sentence_transformers")
        if backend == "sentence_transformers":
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
            )

        assert self.tokenizer is not None

        encoded_batches: List[np.ndarray] = []
        indices = range(0, len(texts), batch_size)
        if show_progress:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            indices = tqdm(indices, total=total_batches, desc=self.model_name)

        for start in indices:
            batch_texts = texts[start : start + batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}

            with torch.inference_mode():
                outputs = self.model(**tokenized)
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                pooled = hidden[:, 0]
                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)

            encoded_batches.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(encoded_batches)

    def encode(
        self,
        texts: Union[str, List[str], List[List[str]]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode strings or segmented documents into dense vectors.
        """
        if self.model is None:
            self.load()

        if batch_size is None:
            batch_size = self.model_spec.get("batch_size", 32)

        if isinstance(texts, str):
            return self._encode_text_batch(
                [texts],
                batch_size=batch_size,
                show_progress=show_progress,
                normalize=normalize,
            )

        if texts and isinstance(texts[0], list):
            flat_texts: List[str] = []
            spans: List[tuple[int, int]] = []

            for document in texts:
                segments = [segment for segment in document if segment]
                if not segments:
                    segments = [""]
                spans.append((len(flat_texts), len(segments)))
                flat_texts.extend(segments)

            segment_embeddings = self._encode_text_batch(
                flat_texts,
                batch_size=batch_size,
                show_progress=show_progress,
                normalize=normalize,
            )

            doc_embeddings: List[np.ndarray] = []
            for start, count in spans:
                pooled = segment_embeddings[start : start + count].mean(axis=0)
                if normalize:
                    norm = np.linalg.norm(pooled)
                    if norm > 0:
                        pooled = pooled / norm
                doc_embeddings.append(pooled.astype(np.float32))

            return np.stack(doc_embeddings)

        return self._encode_text_batch(
            list(texts),
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=normalize,
        )


class CentroidClassifier:
    """MBTI classifier using per-type centroid embeddings and cosine similarity."""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None, batch_size: int = 32):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.batch_size = batch_size
        self.centroids: Optional[Dict[str, np.ndarray]] = None
        self.centroid_matrix: Optional[np.ndarray] = None
        self.type_order: Optional[List[str]] = None

    def fit(
        self,
        texts: Sequence[TextSample],
        labels: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> "CentroidClassifier":
        logger.info("Computing centroid embeddings for each MBTI type...")

        if self.embedding_model.model is None:
            self.embedding_model.load()

        if embeddings is None:
            logger.info("Encoding %s documents...", len(texts))
            embeddings = self.embedding_model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress=True,
            )
        else:
            logger.info("Using %s pre-computed embeddings...", len(embeddings))

        self.centroids = {}
        df = pd.DataFrame({"embedding": list(embeddings), "label": labels})

        for mbti_type in MBTI_TYPES:
            type_embeddings = df[df["label"] == mbti_type]["embedding"].tolist()
            if type_embeddings:
                centroid = np.mean(np.stack(type_embeddings), axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self.centroids[mbti_type] = centroid
                logger.info("  %s: %s samples", mbti_type, len(type_embeddings))
            else:
                logger.warning("  %s: No samples found!", mbti_type)

        self.type_order = list(self.centroids.keys())
        self.centroid_matrix = np.stack([self.centroids[t] for t in self.type_order])
        logger.info("Centroids computed for %s MBTI types.", len(self.centroids))
        return self

    def predict(self, texts: Union[TextSample, Sequence[TextSample]]) -> Union[str, List[str]]:
        if self.centroid_matrix is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)

        query_embeddings = self.embedding_model.encode(
            batch,
            batch_size=self.batch_size,
            show_progress=False,
        )
        similarities = cosine_similarity(query_embeddings, self.centroid_matrix)
        best_indices = np.argmax(similarities, axis=1)
        predictions = [self.type_order[i] for i in best_indices]
        return predictions[0] if single_input else predictions

    def predict_proba(self, texts: Union[TextSample, Sequence[TextSample]]) -> np.ndarray:
        if self.centroid_matrix is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)

        query_embeddings = self.embedding_model.encode(
            batch,
            batch_size=self.batch_size,
            show_progress=False,
        )
        similarities = cosine_similarity(query_embeddings, self.centroid_matrix)
        exp_sim = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
        return exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

    def get_type_labels(self) -> List[str]:
        return self.type_order if self.type_order else []


class EmbeddingLogisticRegression:
    """MBTI classifier using embedding features with logistic regression."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        batch_size: int = 32,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.batch_size = batch_size
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=42,
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(
        self,
        texts: Sequence[TextSample],
        labels: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> "EmbeddingLogisticRegression":
        logger.info("Training Embedding + Logistic Regression classifier...")

        if self.embedding_model.model is None:
            self.embedding_model.load()

        if embeddings is None:
            logger.info("Encoding %s documents...", len(texts))
            X = self.embedding_model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress=True,
            )
        else:
            logger.info("Using %s pre-computed embeddings...", len(embeddings))
            X = embeddings

        y = self.label_encoder.fit_transform(labels)
        logger.info("Training Logistic Regression...")
        self.classifier.fit(X, y)
        self.is_fitted = True
        logger.info("Training complete.")
        return self

    def predict(self, texts: Union[TextSample, Sequence[TextSample]]) -> Union[str, List[str]]:
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)

        X = self.embedding_model.encode(
            batch,
            batch_size=self.batch_size,
            show_progress=False,
        )
        y_pred = self.classifier.predict(X)
        predictions = self.label_encoder.inverse_transform(y_pred)
        return predictions[0] if single_input else list(predictions)

    def predict_proba(self, texts: Union[TextSample, Sequence[TextSample]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        single_input = isinstance(texts, str)
        batch = [texts] if single_input else list(texts)

        X = self.embedding_model.encode(
            batch,
            batch_size=self.batch_size,
            show_progress=False,
        )
        return self.classifier.predict_proba(X)

    def get_type_labels(self) -> List[str]:
        return list(self.label_encoder.classes_) if self.is_fitted else []


class EmbeddingPipeline:
    """Unified pipeline for embedding-based MBTI personality prediction."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        self.model_name = resolve_model_name(model_name)
        model_spec = MODEL_SPECS.get(self.model_name, {})
        self.batch_size = batch_size or model_spec.get("batch_size", 32)
        self.embedding_model = EmbeddingModel(
            self.model_name,
            device=device,
            max_length=max_length or model_spec.get("max_length"),
        )
        self.centroid_clf = CentroidClassifier(self.embedding_model, batch_size=self.batch_size)
        self.lr_clf = EmbeddingLogisticRegression(
            self.embedding_model,
            batch_size=self.batch_size,
        )
        self.is_fitted = False

    def fit(self, texts: Sequence[TextSample], labels: List[str]) -> "EmbeddingPipeline":
        logger.info("=" * 60)
        logger.info("Training Embedding Pipeline")
        logger.info("=" * 60)

        logger.info("Encoding %s documents for shared use...", len(texts))
        embeddings = self.embedding_model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress=True,
        )

        logger.info("\n[1/2] Training Centroid Classifier...")
        self.centroid_clf.fit(texts, labels, embeddings=embeddings)

        logger.info("\n[2/2] Training Logistic Regression Classifier...")
        self.lr_clf.fit(texts, labels, embeddings=embeddings)

        self.is_fitted = True
        logger.info("\n" + "=" * 60)
        logger.info("Embedding Pipeline training complete!")
        logger.info("=" * 60)
        return self

    def predict(
        self,
        texts: Union[TextSample, Sequence[TextSample]],
        method: str = "lr",
    ) -> Union[str, List[str]]:
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if method == "centroid":
            return self.centroid_clf.predict(texts)
        if method == "lr":
            return self.lr_clf.predict(texts)
        raise ValueError(f"Unknown method: {method}. Use 'centroid' or 'lr'.")

    def predict_proba(
        self,
        texts: Union[TextSample, Sequence[TextSample]],
        method: str = "lr",
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if method == "centroid":
            return self.centroid_clf.predict_proba(texts)
        if method == "lr":
            return self.lr_clf.predict_proba(texts)
        raise ValueError(f"Unknown method: {method}. Use 'centroid' or 'lr'.")

    def evaluate(
        self,
        texts: Sequence[TextSample],
        labels: List[str],
        method: str = "lr",
    ) -> Dict[str, Any]:
        predictions = self.predict(texts, method=method)
        acc = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        report = classification_report(labels, predictions, zero_division=0)

        logger.info("[%s] Accuracy: %.4f | Macro-F1: %.4f", method.upper(), acc, macro_f1)
        logger.info("\n%s", report)
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "report": report,
        }

    def encode(self, texts: Union[str, List[str], List[List[str]]]) -> np.ndarray:
        return self.embedding_model.encode(texts, batch_size=self.batch_size)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        centroid_data = {
            "centroids": self.centroid_clf.centroids,
            "centroid_matrix": self.centroid_clf.centroid_matrix,
            "type_order": self.centroid_clf.type_order,
        }
        with open(path / "centroid_clf.pkl", "wb") as handle:
            pickle.dump(centroid_data, handle)

        lr_data = {
            "classifier": self.lr_clf.classifier,
            "label_encoder": self.lr_clf.label_encoder,
        }
        with open(path / "lr_clf.pkl", "wb") as handle:
            pickle.dump(lr_data, handle)

        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_model.embedding_dim,
            "batch_size": self.batch_size,
            "max_length": self.embedding_model.max_length,
        }
        with open(path / "metadata.pkl", "wb") as handle:
            pickle.dump(metadata, handle)

        logger.info("Pipeline saved to %s", path)

    def load(self, path: Union[str, Path]) -> "EmbeddingPipeline":
        path = Path(path)

        with open(path / "metadata.pkl", "rb") as handle:
            metadata = pickle.load(handle)

        self.model_name = metadata["model_name"]
        self.batch_size = metadata.get("batch_size", self.batch_size)
        self.embedding_model = EmbeddingModel(
            self.model_name,
            max_length=metadata.get("max_length"),
        )
        self.embedding_model.load()
        self.embedding_model.embedding_dim = metadata["embedding_dim"]

        with open(path / "centroid_clf.pkl", "rb") as handle:
            centroid_data = pickle.load(handle)
        self.centroid_clf = CentroidClassifier(self.embedding_model, batch_size=self.batch_size)
        self.centroid_clf.centroids = centroid_data["centroids"]
        self.centroid_clf.centroid_matrix = centroid_data["centroid_matrix"]
        self.centroid_clf.type_order = centroid_data["type_order"]

        with open(path / "lr_clf.pkl", "rb") as handle:
            lr_data = pickle.load(handle)
        self.lr_clf = EmbeddingLogisticRegression(
            self.embedding_model,
            batch_size=self.batch_size,
        )
        self.lr_clf.classifier = lr_data["classifier"]
        self.lr_clf.label_encoder = lr_data["label_encoder"]
        self.lr_clf.is_fitted = True

        self.is_fitted = True
        logger.info("Pipeline loaded from %s", path)
        return self


def compute_embeddings_for_dataset(
    texts: Union[List[str], List[List[str]]],
    model_name: str = DEFAULT_MODEL,
    batch_size: Optional[int] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Compute embeddings for a dataset and optionally save them."""
    model = EmbeddingModel(model_name)
    model.load()

    embeddings = model.encode(
        texts,
        batch_size=batch_size or MODEL_SPECS.get(resolve_model_name(model_name), {}).get(
            "batch_size",
            32,
        ),
        show_progress=True,
    )

    if output_path:
        np.save(output_path, embeddings)
        logger.info("Embeddings saved to %s", output_path)

    return embeddings


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model_name: str,
    input_mode: str,
    mask_mbti: bool,
    max_posts: Optional[int],
    min_post_words: int,
    batch_size: Optional[int],
) -> Dict[str, Any]:
    """Run one train/eval experiment and return metrics."""
    resolved_model_name = resolve_model_name(model_name)
    train_docs, train_stats = build_documents(
        train_df,
        input_mode=input_mode,
        mask_mbti=mask_mbti,
        max_posts=max_posts,
        min_post_words=min_post_words,
    )
    test_docs, test_stats = build_documents(
        test_df,
        input_mode=input_mode,
        mask_mbti=mask_mbti,
        max_posts=max_posts,
        min_post_words=min_post_words,
    )

    train_labels = train_df["type"].tolist()
    test_labels = test_df["type"].tolist()

    pipeline = EmbeddingPipeline(
        model_name=resolved_model_name,
        batch_size=batch_size,
    )

    logger.info(
        "Prepared input. Train stats=%s | Test stats=%s",
        train_stats,
        test_stats,
    )

    started = time.perf_counter()
    pipeline.fit(train_docs, train_labels)
    train_seconds = time.perf_counter() - started

    centroid_metrics = pipeline.evaluate(test_docs, test_labels, method="centroid")
    lr_metrics = pipeline.evaluate(test_docs, test_labels, method="lr")

    return {
        "model_name": resolved_model_name,
        "input_mode": input_mode,
        "mask_mbti": mask_mbti,
        "max_posts": max_posts,
        "min_post_words": min_post_words,
        "batch_size": pipeline.batch_size,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "train_seconds": round(train_seconds, 2),
        "centroid": centroid_metrics,
        "lr": lr_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    default_data_dir = Path(__file__).parent.parent / "data" / "processed"
    default_results_path = Path(__file__).parent.parent / "results" / "embedding_results.json"

    parser = argparse.ArgumentParser(description="Run MBTI embedding experiments.")
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--model-name",
        default="minilm",
        help="Model alias or full HF identifier.",
    )
    parser.add_argument(
        "--compare-models",
        nargs="*",
        default=None,
        help="Optional list of model aliases to run sequentially.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["clean_posts", "posts_list_pool"],
        default="posts_list_pool",
    )
    parser.add_argument("--disable-mask-mbti", action="store_true")
    parser.add_argument("--max-posts", type=int, default=16)
    parser.add_argument("--min-post-words", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--results-path", default=str(default_results_path))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_df = load_split_dataframe(data_dir, args.train_split)
    test_df = load_split_dataframe(data_dir, args.test_split)

    model_names = args.compare_models or [args.model_name]
    results = []

    for model_name in model_names:
        logger.info("=" * 80)
        logger.info(
            "Running experiment | model=%s | input_mode=%s | mask_mbti=%s",
            model_name,
            args.input_mode,
            not args.disable_mask_mbti,
        )
        logger.info("=" * 80)
        experiment_result = run_experiment(
            train_df,
            test_df,
            model_name=model_name,
            input_mode=args.input_mode,
            mask_mbti=not args.disable_mask_mbti,
            max_posts=args.max_posts,
            min_post_words=args.min_post_words,
            batch_size=args.batch_size,
        )
        results.append(experiment_result)

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("=" * 80)
    print("Experiment summary")
    print("=" * 80)
    for result in results:
        print(
            f"{result['model_name']} | {result['input_mode']} | "
            f"centroid acc={result['centroid']['accuracy']:.4f} "
            f"macro_f1={result['centroid']['macro_f1']:.4f} | "
            f"lr acc={result['lr']['accuracy']:.4f} "
            f"macro_f1={result['lr']['macro_f1']:.4f} | "
            f"train_seconds={result['train_seconds']:.2f}"
        )
    print(f"Saved detailed results to {results_path}")


if __name__ == "__main__":
    main()
