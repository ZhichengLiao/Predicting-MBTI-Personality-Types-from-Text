# MBTI Personality Type Prediction

CS410 project for predicting MBTI personality type from text.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Project Check

```bash
python run_project.py
```

This loads the saved TF-IDF + Logistic Regression dichotomy model and evaluates it on `data/processed/test.csv`.

## Run the Web App

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL shown by Streamlit. The app uses the trained model at:

```text
results/dichotomy_classifiers.pkl
```

## Online Demo

Try the deployed app here:

https://mbti-by-perfectiveai.streamlit.app/

This demo is hosted on Streamlit Community Cloud (`streamlit.app`) from a GitHub repository.

## Optional: Retrain the Dichotomy Model

```bash
python run_project.py --train
```

This expects the processed data in `data/processed/` and writes the trained model to `results/`.

## Key Files

- `app/streamlit_app.py`: Streamlit app entry point
- `app/pages/Advanced_Mode.py`: advanced analysis page
- `models/dichotomy_classifiers.py`: four binary classifiers for E/I, S/N, T/F, J/P
- `models/embedding.py`: separate embedding experiments, not used by the deployed app
- `run_project.py`: automatic evaluation script
- `requirements.txt`: Python dependencies
