"""Advanced Mode — four interactive views over the same trained model.

  Live      — prediction updates as you write
  Signals   — words highlighted by which axis they pull toward
  Refine    — append more text and watch each axis shift
  Compare   — two passages side by side, deltas surfaced
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))
from models.dichotomy_classifiers import DichotomyClassifiers, DIM_LABELS  # noqa: E402

# Shared loader — same cache key as the simple page, so the model
# stays warm when navigating between pages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import load_model  # noqa: E402

# ---------------------------------------------------------------------------
# Cleaning, profiles
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MBTI_RE = re.compile(
    r"\b(?:intj|intp|entj|entp|infj|infp|enfj|enfp|"
    r"istj|isfj|estj|esfj|istp|isfp|estp|esfp)s?\b",
    re.IGNORECASE,
)
NON_ALPHA_RE = re.compile(r"[^a-z\s]+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MBTI_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


AXIS_NAMES = {
    "E/I": {"E": "Extroverted", "I": "Introverted"},
    "S/N": {"S": "Sensing",     "N": "Intuitive"},
    "T/F": {"T": "Thinking",    "F": "Feeling"},
    "J/P": {"J": "Judging",     "P": "Perceiving"},
}

# One color per axis. Used for inline highlights and bars.
AXIS_COLOR = {
    "E/I": "#F97316",  # orange
    "S/N": "#7C3AED",  # purple
    "T/F": "#DB2777",  # pink
    "J/P": "#06B6D4",  # cyan
}

NICKNAMES = {
    "INTJ": "the Architect", "INTP": "the Logician",   "ENTJ": "the Commander",  "ENTP": "the Debater",
    "INFJ": "the Advocate",  "INFP": "the Mediator",   "ENFJ": "the Protagonist","ENFP": "the Campaigner",
    "ISTJ": "the Logistician","ISFJ": "the Defender",  "ESTJ": "the Executive",  "ESFJ": "the Consul",
    "ISTP": "the Virtuoso",  "ISFP": "the Adventurer", "ESTP": "the Entrepreneur","ESFP": "the Entertainer",
}


def predict_full(clf, raw: str):
    """Return (cleaned, mbti, axis_probs) or None if input is empty post-cleaning."""
    cleaned = clean_text(raw)
    if not cleaned:
        return None
    mbti = clf.predict(cleaned)
    probs = clf.predict_proba(cleaned)[0]
    return cleaned, mbti, probs


def axis_winner(clf, dim: int, p1: float):
    """For a given axis, return (winning_letter, winning_prob, full_name)."""
    cls0, cls1 = clf.encoders[dim].classes_
    if p1 >= 0.5:
        return cls1, p1, AXIS_NAMES[DIM_LABELS[dim]][cls1]
    return cls0, 1 - p1, AXIS_NAMES[DIM_LABELS[dim]][cls0]


def render_type_card(mbti: str) -> str:
    nick = NICKNAMES.get(mbti, "")
    return (
        f'<div style="background:linear-gradient(135deg,#1E3A8A,#2563EB);'
        f'color:white;padding:1.2rem 1.4rem;border-radius:14px;'
        f'box-shadow:0 6px 20px rgba(30,58,138,0.18);">'
        f'<div style="font-size:2.6rem;font-weight:800;letter-spacing:0.08em;'
        f'line-height:1;">{mbti}</div>'
        f'<div style="opacity:0.92;margin-top:0.25rem;">{nick}</div></div>'
    )


def render_axis_bars(clf, probs, highlight_letters: dict | None = None) -> str:
    """Vertical stack of axis bars. If highlight_letters supplied (dim->letter),
    pulse the bar to indicate that this axis flipped vs a baseline."""
    parts = []
    for dim in range(4):
        axis = DIM_LABELS[dim]
        letter, p, full = axis_winner(clf, dim, float(probs[dim]))
        col = AXIS_COLOR[axis]
        flag = ""
        if highlight_letters and highlight_letters.get(dim) == "flipped":
            flag = (
                ' <span style="font-size:0.72rem;background:#FEE2E2;color:#991B1B;'
                'padding:1px 6px;border-radius:6px;font-weight:700;">flipped</span>'
            )
        parts.append(
            f'<div style="margin:8px 0;font-size:0.92rem;">'
            f'<b>{axis}</b> <span style="color:#475569">·</span> {full}{flag}'
            f'<span style="float:right;color:#64748B;font-weight:600">{p:.0%}</span>'
            f'<div style="height:7px;background:#E2E8F0;border-radius:99px;'
            f'margin-top:4px;overflow:hidden;">'
            f'<div style="width:{p*100:.1f}%;height:100%;background:{col};'
            f'border-radius:99px;"></div></div></div>'
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Inline word highlighting
# ---------------------------------------------------------------------------


def build_token_contributions(clf, cleaned: str):
    """For each unigram token in `cleaned`, find its strongest signed
    contribution across the 4 axes (TF-IDF × LR coef). Return a dict
    {token_lower: (dim, signed_value, target_letter)} for the dominant axis."""
    if not cleaned:
        return {}

    contribs: dict[str, tuple[int, float, str]] = {}
    for dim in range(4):
        vec = clf.vectorizers[dim]
        lr = clf.classifiers[dim]
        encoder = clf.encoders[dim]
        X = vec.transform([cleaned])
        if X.nnz == 0:
            continue
        feature_names = vec.get_feature_names_out()
        coef = lr.coef_[0]
        for col_idx, tfidf_val in zip(X.indices, X.data):
            feat = feature_names[col_idx]
            if " " in feat:
                continue  # skip bigrams — only colorize single words
            signed = float(tfidf_val * coef[col_idx])
            target = encoder.classes_[1] if signed > 0 else encoder.classes_[0]
            existing = contribs.get(feat)
            if existing is None or abs(signed) > abs(existing[1]):
                contribs[feat] = (dim, signed, target)
    return contribs


def render_highlighted_text(raw_text: str, contribs: dict, threshold: float = 0.005) -> str:
    """Wrap impactful tokens in colored spans. Preserves whitespace and punctuation."""
    if not raw_text.strip():
        return ""

    parts = []
    for tok in re.findall(r"[A-Za-z]+|[^A-Za-z]+", raw_text):
        if not tok or not tok[0].isalpha():
            parts.append(html.escape(tok))
            continue
        info = contribs.get(tok.lower())
        if info is None or abs(info[1]) < threshold:
            parts.append(html.escape(tok))
            continue
        dim, signed, target_letter = info
        axis = DIM_LABELS[dim]
        col = AXIS_COLOR[axis]
        # Opacity scales with magnitude (capped). Direction shown via solid vs dashed underline.
        opacity = min(0.55, 0.18 + abs(signed) * 4)
        underline = "solid" if signed > 0 else "dashed"
        tooltip = f"{axis} → {target_letter} ({signed:+.3f})"
        parts.append(
            f'<span title="{html.escape(tooltip)}" '
            f'style="background:{col}{int(opacity*255):02X};'
            f'border-bottom:2px {underline} {col};'
            f'border-radius:3px;padding:0 2px;">{html.escape(tok)}</span>'
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Persona · Advanced",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    .mode-switcher {
        display: inline-flex; padding: 4px; gap: 4px;
        background: #F1F5F9; border-radius: 999px;
        margin: 0.25rem 0 1.2rem 0;
    }
    .mode-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 9px 22px; border-radius: 999px;
        font-size: 0.92rem; font-weight: 600;
        color: #64748B; text-decoration: none;
        transition: color 160ms ease, background 160ms ease;
    }
    .mode-pill:hover { color: #1E3A8A; }
    .mode-pill.active {
        background: white; color: #1E3A8A;
        box-shadow: 0 2px 6px rgba(30,58,138,0.10);
    }
    .mode-pill .accent { color: #F97316; }
    .page-title {
        font-size: 2rem; font-weight: 800; color: #1E3A8A;
        margin: 0.5rem 0 0.25rem 0;
    }
    .page-sub {
        color: #64748B; font-size: 0.98rem; margin-bottom: 1.2rem;
    }
    .legend-row { display: flex; gap: 16px; flex-wrap: wrap;
                  font-size: 0.82rem; color: #475569; margin: 6px 0 14px; }
    .legend-chip {
        display: inline-flex; align-items: center; gap: 6px;
    }
    .legend-swatch {
        width: 14px; height: 14px; border-radius: 4px;
    }
    .stButton > button {
        background: #F97316; color: white; border: none;
        font-weight: 600; border-radius: 10px; padding: 0.5rem 1.1rem;
    }
    .stButton > button:hover { background: #EA580C; color: white; }
    .highlighted-text {
        background: white; padding: 1rem 1.2rem; border-radius: 12px;
        line-height: 1.85; font-size: 1rem; color: #0F172A;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">Advanced Mode</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Four ways to interrogate the model with your own writing.</div>',
            unsafe_allow_html=True)

# Mode switcher — same component as the simple page, with Advanced active.
# Inline onclick handler navigates via the top window so the link works
# reliably whether or not Streamlit serves the app inside an iframe.
st.markdown(
    '<div class="mode-switcher">'
    '<a class="mode-pill" href="/" '
    'onclick="(window.top||window).location.href=\'/\';return false;">'
    '<span class="accent">←</span> Simple</a>'
    '<span class="mode-pill active">Advanced</span>'
    '</div>',
    unsafe_allow_html=True,
)

clf = load_model()

tab_live, tab_signals, tab_refine, tab_compare = st.tabs(
    ["🔴  Live", "🎨  Word signals", "🔄  Refine", "⚖️  Compare"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Live prediction (re-runs on each text update)
# ---------------------------------------------------------------------------
with tab_live:
    st.caption("Type below, then click Analyze. Your reading also auto-updates whenever you click outside the text box.")
    col_in, col_out = st.columns([1.1, 1])
    with col_in:
        live_text = st.text_area(
            "Your text",
            key="live_text",
            height=280,
            placeholder="Start typing — a few sentences works best.",
            label_visibility="collapsed",
        )
        st.button("Analyze", key="live_go", type="primary", use_container_width=True)
    with col_out:
        if not live_text or len(live_text.strip()) < 30:
            st.info("Write at least a few sentences to see a reading.")
        else:
            result = predict_full(clf, live_text)
            if result is None:
                st.warning("Nothing to read after cleanup — try more text.")
            else:
                _, mbti, probs = result
                st.markdown(render_type_card(mbti), unsafe_allow_html=True)
                st.markdown(render_axis_bars(clf, probs), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2 — Inline word highlighting
# ---------------------------------------------------------------------------
with tab_signals:
    st.caption(
        "Each word in your text is colored by the axis it pulls toward. "
        "Solid underline = pulls toward the second letter on that axis "
        "(I, N, F, P); dashed underline = pulls toward the first (E, S, T, J). "
        "Hover any word to see the magnitude."
    )

    # Legend
    legend = "".join(
        f'<span class="legend-chip"><span class="legend-swatch" '
        f'style="background:{AXIS_COLOR[axis]};"></span>{axis}</span>'
        for axis in DIM_LABELS
    )
    st.markdown(f'<div class="legend-row">{legend}</div>', unsafe_allow_html=True)

    sig_text = st.text_area(
        "Your text",
        key="signals_text",
        height=180,
        placeholder="Paste a few sentences here.",
        label_visibility="collapsed",
    )
    st.button("Analyze", key="signals_go", type="primary", use_container_width=True)
    if sig_text and len(sig_text.strip()) >= 30:
        result = predict_full(clf, sig_text)
        if result:
            cleaned, mbti, probs = result
            contribs = build_token_contributions(clf, cleaned)
            highlighted = render_highlighted_text(sig_text, contribs)
            st.markdown(f'<div class="highlighted-text">{highlighted}</div>',
                        unsafe_allow_html=True)
            col_a, col_b = st.columns([1, 1.3])
            with col_a:
                st.markdown(render_type_card(mbti), unsafe_allow_html=True)
            with col_b:
                st.markdown(render_axis_bars(clf, probs), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 3 — Refine: append text and watch the prediction shift
# ---------------------------------------------------------------------------
with tab_refine:
    st.caption(
        "Start with a seed passage. Then add more — the new prediction is "
        "computed on the combined text. See which axes hold and which flip."
    )
    base_text = st.text_area(
        "Seed text",
        key="refine_base",
        height=140,
        placeholder="Your starting passage…",
    )
    add_text = st.text_area(
        "Add more",
        key="refine_add",
        height=120,
        placeholder="Append a continuation, an unrelated thought, anything…",
    )
    st.button("Analyze", key="refine_go", type="primary", use_container_width=True)

    if base_text and len(base_text.strip()) >= 30:
        base_result = predict_full(clf, base_text)
        if base_result:
            _, base_mbti, base_probs = base_result
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Before**")
                st.markdown(render_type_card(base_mbti), unsafe_allow_html=True)
                st.markdown(render_axis_bars(clf, base_probs), unsafe_allow_html=True)

            if add_text and len(add_text.strip()) >= 5:
                combined = base_text.strip() + " " + add_text.strip()
                new_result = predict_full(clf, combined)
                if new_result:
                    _, new_mbti, new_probs = new_result
                    flipped = {
                        dim: ("flipped" if (base_probs[dim] >= 0.5) != (new_probs[dim] >= 0.5) else None)
                        for dim in range(4)
                    }
                    with col_r:
                        st.markdown("**After**")
                        st.markdown(render_type_card(new_mbti), unsafe_allow_html=True)
                        st.markdown(render_axis_bars(clf, new_probs, flipped),
                                    unsafe_allow_html=True)

                    deltas = []
                    for dim in range(4):
                        delta = float(new_probs[dim] - base_probs[dim])
                        deltas.append((DIM_LABELS[dim], delta))
                    delta_strs = " · ".join(
                        f'<b>{ax}</b> <span style="color:{"#16A34A" if d > 0 else "#DC2626"}">'
                        f'{d:+.2f}</span>'
                        for ax, d in deltas
                    )
                    st.markdown(
                        f'<div style="margin-top:14px;font-size:0.9rem;color:#475569;">'
                        f'Δ probability shift per axis: {delta_strs}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                with col_r:
                    st.info("Add a continuation in the second box to see the shift.")

# ---------------------------------------------------------------------------
# Tab 4 — Compare two passages
# ---------------------------------------------------------------------------
with tab_compare:
    st.caption("Two passages, side by side. Axes that disagree get flagged.")
    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area(
            "Voice A",
            key="cmp_a",
            height=240,
            placeholder="First passage — e.g. how you write to friends.",
        )
    with col_b:
        text_b = st.text_area(
            "Voice B",
            key="cmp_b",
            height=240,
            placeholder="Second passage — e.g. how you write at work.",
        )
    st.button("Analyze", key="compare_go", type="primary", use_container_width=True)

    a_ok = text_a and len(text_a.strip()) >= 30
    b_ok = text_b and len(text_b.strip()) >= 30
    if a_ok and b_ok:
        a_result = predict_full(clf, text_a)
        b_result = predict_full(clf, text_b)
        if a_result and b_result:
            _, a_mbti, a_probs = a_result
            _, b_mbti, b_probs = b_result

            disagreed = {
                dim: ("flipped" if (a_probs[dim] >= 0.5) != (b_probs[dim] >= 0.5) else None)
                for dim in range(4)
            }

            with col_a:
                st.markdown(render_type_card(a_mbti), unsafe_allow_html=True)
                st.markdown(render_axis_bars(clf, a_probs, disagreed), unsafe_allow_html=True)
            with col_b:
                st.markdown(render_type_card(b_mbti), unsafe_allow_html=True)
                st.markdown(render_axis_bars(clf, b_probs, disagreed), unsafe_allow_html=True)

            differing = [DIM_LABELS[d] for d, v in disagreed.items() if v == "flipped"]
            if differing:
                st.markdown(
                    f'<div style="margin-top:18px;padding:12px 16px;background:#FEF3C7;'
                    f'border-radius:10px;color:#92400E;font-size:0.95rem;">'
                    f'<b>Voice difference:</b> these axes disagreed across the two passages: '
                    f'{", ".join(differing)}.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="margin-top:18px;padding:12px 16px;background:#DCFCE7;'
                    f'border-radius:10px;color:#166534;font-size:0.95rem;">'
                    f'Both passages read as the same MBTI type — your voice is consistent.</div>',
                    unsafe_allow_html=True,
                )
    elif text_a or text_b:
        st.info("Both passages need a few sentences to compare.")

st.markdown(
    '<div style="color:#94A3B8;font-size:0.78rem;text-align:center;margin-top:3rem;">'
    'For exploration and entertainment — not a clinical assessment.</div>',
    unsafe_allow_html=True,
)
