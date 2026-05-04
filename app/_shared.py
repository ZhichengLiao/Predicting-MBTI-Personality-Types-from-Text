"""Shared helpers for the Persona MBTI app.

The whole point of this module: define `load_model` once, so both
pages (Simple and Advanced) share the same Streamlit cache entry for
the trained classifier. When the user navigates between pages, the
model stays warm — no redundant unpickling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))
from models.dichotomy_classifiers import DichotomyClassifiers  # noqa: E402

RESULTS_DIR = PROJ_ROOT / "results"


@st.cache_resource(show_spinner="Warming up…")
def load_model() -> DichotomyClassifiers:
    clf = DichotomyClassifiers()
    clf.load(RESULTS_DIR)
    return clf
