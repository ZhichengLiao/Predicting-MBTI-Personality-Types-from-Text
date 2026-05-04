"""Persona — a personality-aware AI assistant."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))
from models.dichotomy_classifiers import DichotomyClassifiers, DIM_LABELS  # noqa: E402

# Single source of truth for `load_model` — its cache survives page navigation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import load_model  # noqa: E402

# ---------------------------------------------------------------------------
# Text cleanup
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


def signal_words(clf: DichotomyClassifiers, dim: int, cleaned_text: str, k: int = 4):
    """Top words from the user's text on each side of one axis."""
    vec = clf.vectorizers[dim]
    lr = clf.classifiers[dim]
    encoder = clf.encoders[dim]

    X = vec.transform([cleaned_text])
    if X.nnz == 0:
        return [(encoder.classes_[1], []), (encoder.classes_[0], [])]

    coef = lr.coef_[0]
    feat_names = vec.get_feature_names_out()
    cols = X.indices
    contrib = X.data * coef[cols]

    order = np.argsort(contrib)
    pos_words = [feat_names[cols[i]] for i in order[::-1][:k] if contrib[i] > 0]
    neg_words = [feat_names[cols[i]] for i in order[:k] if contrib[i] < 0]
    return [(encoder.classes_[1], pos_words), (encoder.classes_[0], neg_words)]


# ---------------------------------------------------------------------------
# Personality profiles
# ---------------------------------------------------------------------------

TYPE_PROFILES: dict[str, dict] = {
    "INTJ": {"nickname": "the Architect",   "vibe": "strategic, independent, future-focused",
             "topics": ["systems design", "long-term planning", "philosophy of mind", "chess", "hard sci-fi"]},
    "INTP": {"nickname": "the Logician",    "vibe": "analytical, curious, theory-loving",
             "topics": ["theoretical CS", "epistemology", "puzzles", "open-source", "math paradoxes"]},
    "ENTJ": {"nickname": "the Commander",   "vibe": "decisive, organized, goal-driven",
             "topics": ["leadership", "competitive strategy", "career growth", "negotiation", "macroeconomics"]},
    "ENTP": {"nickname": "the Debater",     "vibe": "inventive, witty, idea-juggling",
             "topics": ["startup ideas", "rhetoric", "futurism", "improv", "contrarian essays"]},
    "INFJ": {"nickname": "the Advocate",    "vibe": "insightful, idealistic, quietly intense",
             "topics": ["narrative therapy", "social impact", "depth psychology", "literary fiction", "ethics"]},
    "INFP": {"nickname": "the Mediator",    "vibe": "empathetic, imaginative, value-driven",
             "topics": ["creative writing", "indie music", "journaling", "fantasy worldbuilding", "self-discovery"]},
    "ENFJ": {"nickname": "the Protagonist", "vibe": "warm, encouraging, people-focused",
             "topics": ["mentorship", "community organizing", "public speaking", "education", "team rituals"]},
    "ENFP": {"nickname": "the Campaigner",  "vibe": "enthusiastic, exploratory, spontaneous",
             "topics": ["travel stories", "side projects", "creative collabs", "personality theory", "festivals"]},
    "ISTJ": {"nickname": "the Logistician", "vibe": "thorough, dependable, tradition-respecting",
             "topics": ["personal finance", "history", "process improvement", "home organization", "DIY repair"]},
    "ISFJ": {"nickname": "the Defender",    "vibe": "supportive, conscientious, detail-aware",
             "topics": ["caregiving", "cozy hobbies", "family traditions", "wellness", "small-group volunteering"]},
    "ESTJ": {"nickname": "the Executive",   "vibe": "practical, organized, results-oriented",
             "topics": ["project management", "civic engagement", "coaching", "operations", "law"]},
    "ESFJ": {"nickname": "the Consul",      "vibe": "sociable, harmonious, caring",
             "topics": ["event planning", "hospitality", "community service", "cooking", "PTA life"]},
    "ISTP": {"nickname": "the Virtuoso",    "vibe": "hands-on, observant, troubleshooter",
             "topics": ["tinkering", "motorsports", "embedded electronics", "outdoor survival", "speedruns"]},
    "ISFP": {"nickname": "the Adventurer",  "vibe": "gentle, aesthetic, present-moment",
             "topics": ["visual art", "photography walks", "music", "small kindnesses", "minimalism"]},
    "ESTP": {"nickname": "the Entrepreneur","vibe": "energetic, pragmatic, action-first",
             "topics": ["sports", "sales tactics", "extreme hobbies", "real estate", "live entertainment"]},
    "ESFP": {"nickname": "the Entertainer", "vibe": "playful, expressive, crowd-loving",
             "topics": ["performing arts", "social events", "fashion", "food trends", "travel vlogging"]},
}

AXIS_NAMES = {
    "E/I": {"E": "Extroverted", "I": "Introverted"},
    "S/N": {"S": "Sensing",     "N": "Intuitive"},
    "T/F": {"T": "Thinking",    "F": "Feeling"},
    "J/P": {"J": "Judging",     "P": "Perceiving"},
}


def tailored_response(mbti: str, axis_probs: np.ndarray) -> str:
    profile = TYPE_PROFILES.get(mbti, {"nickname": "your type", "vibe": "uniquely yourself"})
    confidences = np.abs(axis_probs - 0.5)
    strongest_dim = int(np.argmax(confidences))
    strongest_label = DIM_LABELS[strongest_dim]
    strongest_conf = float(confidences[strongest_dim] * 2)

    return (
        f"Reading your words, you come across as **{profile['nickname']}** — "
        f"{profile['vibe']}.\n\n"
        f"Your **{strongest_label}** dimension showed up most distinctly "
        f"(confidence ≈ {strongest_conf:.0%}). When I respond to you, I'll be "
        f"{'direct and structured' if mbti[0] == 'E' else 'thoughtful and unhurried'}, "
        f"lean on {'concrete examples' if mbti[1] == 'S' else 'big-picture patterns'}, "
        f"weigh {'logical tradeoffs' if mbti[2] == 'T' else 'how it feels to people involved'}, "
        f"and offer {'clear next steps' if mbti[3] == 'J' else 'open-ended options'}."
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Persona · personality-aware AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Orange (accent) + blue (primary) palette, sparse custom CSS.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .hero-title {
        font-size: 2.6rem; font-weight: 800; color: #1E3A8A;
        letter-spacing: -0.02em; margin: 0;
    }
    .hero-accent { color: #F97316; }
    .hero-sub {
        color: #475569; font-size: 1.05rem; margin: 0.25rem 0 1.75rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white; padding: 1.6rem 1.8rem; border-radius: 14px;
        box-shadow: 0 6px 24px rgba(30,58,138,0.18);
    }
    .type-letters {
        font-size: 3.2rem; font-weight: 800; line-height: 1;
        letter-spacing: 0.08em;
    }
    .type-nickname { font-size: 1.05rem; opacity: 0.92; margin-top: 0.35rem; }
    .type-vibe     { font-size: 0.92rem; opacity: 0.78; margin-top: 0.5rem; }
    .axis-label    { font-size: 0.82rem; color: #475569; }
    .axis-value    { font-size: 0.95rem; font-weight: 600; color: #1E3A8A; }
    .bar-track {
        height: 8px; background: #E2E8F0; border-radius: 999px; overflow: hidden;
        margin: 4px 0 12px 0;
    }
    .bar-fill {
        height: 100%; background: linear-gradient(90deg, #1E3A8A, #F97316);
        border-radius: 999px;
    }
    .chip {
        display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
        border-radius: 999px; font-size: 0.85rem; font-weight: 500;
    }
    .chip-topic  { background: #FFEDD5; color: #C2410C; }
    .chip-signal { background: #DBEAFE; color: #1E3A8A; }
    .panel-h     { font-size: 0.95rem; font-weight: 700; color: #1E3A8A;
                   margin: 1.4rem 0 0.5rem 0; }
    .axis-mini   { font-size: 0.78rem; color: #64748B; margin-top: 0.6rem; }
    .footer-note { color: #94A3B8; font-size: 0.78rem; text-align: center;
                   margin-top: 3rem; }
    .stButton > button {
        background: #F97316; color: white; border: none;
        font-weight: 600; border-radius: 10px; padding: 0.55rem 1.2rem;
    }
    .stButton > button:hover { background: #EA580C; color: white; }
    /* Hide Streamlit's auto-generated multipage sidebar so only our explicit nav shows. */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    /* Mode switcher pills (Simple / Advanced) */
    .mode-switcher {
        display: inline-flex; padding: 4px; gap: 4px;
        background: #F1F5F9; border-radius: 999px;
        margin: 0.25rem 0 1.5rem 0;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Persona <span class="hero-accent">·</span> personality-aware AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Share a few sentences in your own voice. We\'ll read between the lines '
    'and tailor what we say back to you.</div>',
    unsafe_allow_html=True,
)

# Mode switcher — places the choice front-and-center under the hero.
# Uses an inline onclick handler so navigation works reliably whether or not
# Streamlit Cloud serves the app inside an iframe.
st.markdown(
    '<div class="mode-switcher">'
    '<span class="mode-pill active">Simple</span>'
    '<a class="mode-pill" href="/Advanced_Mode" '
    'onclick="(window.top||window).location.href=\'/Advanced_Mode\';return false;">'
    'Advanced <span class="accent">→</span></a>'
    '</div>',
    unsafe_allow_html=True,
)

clf = load_model()

EXAMPLES = {
    "— pick a sample —": "",
    "On trust and ideas":
        "It depends on what the environment is. I'm very cautious when it comes to "
        "trusting people, I expect most people to have an ulterior motive. But when "
        "it comes to exploring an idea, I'm not cautious at all — I'll chase it down "
        "wherever it leads, because being wrong about a theory feels much less costly "
        "than being wrong about a person.",
    "Hiding behind 'I'm good'":
        "If anyone asks how my day was or how I am in general, I usually stick to "
        "'Good', regardless if it's true or not. It's a positive response that rarely "
        "elicits a request for elaboration. My actual day could be falling apart and "
        "I'd still go with Good, because the alternative is a conversation I don't "
        "really want to have.",
    "Comforting a stranger online":
        "I'm horrible with this kind of thing, but I feel bad that no one has said "
        "anything yet, so I'll try. I used to be in this same situation, with my first "
        "love of a year, and like you said it just broke something in me for a while. "
        "What helped was letting myself feel it instead of pretending I was fine.",
    "Returning after a long absence":
        "Hey guys. I know it's been a long time since I've been active on here. Needed "
        "to continue this work in the seclusion of my Batcave for a while. I hope to "
        "be on a little bit more in coming weeks. There's a lot I've been turning over "
        "quietly that I'd like to start putting into words again.",
    "Excited about rewatching a film":
        "Wow! Amazing… I totally agree with you! I also rewatched the trilogy and "
        "got the same feeling about Morpheus. I couldn't have explained it better than "
        "you. The whole thing hit different this time around — I kept pausing to point "
        "stuff out to my roommate even though she's seen it a hundred times. Awesome!",
    "Playful imagery / wordplay":
        "When I read this line I imagined a Drunk Parrot riding atop a Wrecking ball "
        "with the word Truth emblazoned on the side a la Miley Cyrus. Which honestly "
        "raises more questions than it answers, but I'm choosing to commit to the bit "
        "rather than examine why my brain went there first.",
    "Difficulty opening up":
        "I'm not one to open up, but I do have feelings. I am just one to feel very "
        "uncomfortable about letting others know what they are. He may want to know "
        "this is actually going somewhere, so sometime in the near future I'll have "
        "to find a way to say it — even if it comes out clumsy.",
    "Cozy movie recommendation":
        "The character that I probably relate to the most is Sophie from Howl's Moving "
        "Castle. Amazing movie by the way — if you haven't seen it, go watch it online "
        "now, or any Studio Ghibli film for that matter. They're the kind of movies "
        "that just feel like a warm blanket on a rainy afternoon.",
    "Single and loving it":
        "Single. Much friends, much fun, nothing to regret — and interestingly, my "
        "female friends keep introducing me to their friends, and even more friends, "
        "so more fun to come. Honestly I'm not looking for anything serious right now; "
        "the energy of a packed weekend is doing plenty for me.",
}

col_left, col_right = st.columns([1.05, 1])

with col_left:
    preset = st.selectbox("Try a sample", list(EXAMPLES.keys()))
    user_text = st.text_area(
        "Or write your own — a few sentences works best",
        value=EXAMPLES[preset],
        height=240,
        placeholder="What's on your mind?",
        label_visibility="visible",
    )
    go = st.button("Analyze", type="primary", use_container_width=True)

with col_right:
    if not (go and user_text.strip()):
        st.markdown(
            """
            <div style="background:#F1F5F9;border-radius:14px;padding:2.4rem 1.6rem;
                        text-align:center;color:#64748B;height:100%;">
              <div style="font-size:2.5rem;margin-bottom:0.6rem;">✨</div>
              <div style="font-weight:600;color:#1E3A8A;">Your reading appears here</div>
              <div style="font-size:0.9rem;margin-top:0.4rem;">
                Pick a sample on the left, or paste anything you've written recently.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        cleaned = clean_text(user_text)
        if not cleaned:
            st.warning("There wasn't enough text to read. Try a few full sentences.")
        else:
            mbti = clf.predict(cleaned)
            probs = clf.predict_proba(cleaned)[0]
            profile = TYPE_PROFILES.get(mbti, {})

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="type-letters">{mbti}</div>
                  <div class="type-nickname">{profile.get('nickname','')}</div>
                  <div class="type-vibe">{profile.get('vibe','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for dim in range(4):
                p1 = float(probs[dim])
                cls0, cls1 = clf.encoders[dim].classes_
                winner_letter, winner_p = (cls1, p1) if p1 >= 0.5 else (cls0, 1 - p1)
                axis = DIM_LABELS[dim]
                full = AXIS_NAMES[axis][winner_letter]
                st.markdown(
                    f'<div class="axis-label"><b>{axis}</b> · '
                    f'<span class="axis-value">{full}</span> '
                    f'<span style="float:right;color:#64748B;">{winner_p:.0%}</span></div>'
                    f'<div class="bar-track"><div class="bar-fill" '
                    f'style="width:{winner_p*100:.1f}%"></div></div>',
                    unsafe_allow_html=True,
                )

# Below-the-fold: personalized take + topics + key signals
if go and user_text.strip():
    cleaned = clean_text(user_text)
    if cleaned:
        st.divider()
        col_a, col_b = st.columns([1.3, 1])

        with col_a:
            st.markdown('<div class="panel-h">Your personalized take</div>', unsafe_allow_html=True)
            st.markdown(tailored_response(mbti, probs))

            if profile.get("topics"):
                st.markdown('<div class="panel-h">Topics you might enjoy</div>', unsafe_allow_html=True)
                chips = "".join(f'<span class="chip chip-topic">{t}</span>' for t in profile["topics"])
                st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="panel-h">Key signals from your text</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="color:#64748B;font-size:0.85rem;margin-bottom:0.6rem;">'
                'Words from your writing that pointed most clearly to each side.</div>',
                unsafe_allow_html=True,
            )
            for dim in range(4):
                axis = DIM_LABELS[dim]
                buckets = signal_words(clf, dim, cleaned, k=4)
                rendered_any = False
                row = f'<div class="axis-mini"><b>{axis}</b></div>'
                for letter, words in buckets:
                    if not words:
                        continue
                    full = AXIS_NAMES[axis][letter]
                    chips = "".join(f'<span class="chip chip-signal">{w}</span>' for w in words)
                    row += f'<div style="margin:4px 0 6px 0;font-size:0.78rem;color:#64748B;">→ {full}</div>{chips}'
                    rendered_any = True
                if rendered_any:
                    st.markdown(row, unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">For exploration and entertainment — not a clinical assessment.</div>',
    unsafe_allow_html=True,
)
