from pathlib import Path
import re

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack

ASPECTS = ["food", "service", "price", "cleanliness", "delivery", "ambiance", "app_experience", "general"]
LABEL_TO_SENTIMENT = {1: "positive", 2: "negative"}


class PredictRequest(BaseModel):
    review_text: str


def remove_tashkeel(text: str) -> str:
    tashkeel = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    return re.sub(tashkeel, "", text)


def normalize_arabic(text: str) -> str:
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    return text


def remove_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def clean_text(text: str) -> str:
    text = str(text)
    text = remove_tashkeel(text)
    text = normalize_arabic(text)
    text = remove_repeated_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_model_text(cleaned_text: str) -> str:
    # Keep the same training text pattern: clean_text + business_category + platform + star_rating
    return f"{cleaned_text}   "


def load_artifacts() -> tuple:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    svm_path = models_dir / "models_svm.pkl"
    word_vec_path = models_dir / "word_vectorizer.pkl"
    char_vec_path = models_dir / "char_vectorizer.pkl"

    missing = [str(p) for p in [svm_path, word_vec_path, char_vec_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    models_svm = joblib.load(svm_path)
    word_vectorizer = joblib.load(word_vec_path)
    char_vectorizer = joblib.load(char_vec_path)
    return models_svm, word_vectorizer, char_vectorizer


app = FastAPI(title="Arabic ABSA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    MODELS_SVM, WORD_VECTORIZER, CHAR_VECTORIZER = load_artifacts()
    MODEL_LOAD_ERROR = None
except Exception as exc:
    MODELS_SVM = None
    WORD_VECTORIZER = None
    CHAR_VECTORIZER = None
    MODEL_LOAD_ERROR = str(exc)


@app.get("/")
def health_check() -> dict:
    return {"status": "ok", "message": "Arabic ABSA API is running"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if MODEL_LOAD_ERROR:
        raise HTTPException(status_code=500, detail=f"Model loading error: {MODEL_LOAD_ERROR}")

    cleaned = clean_text(payload.review_text)
    model_text = build_model_text(cleaned)

    x_word = WORD_VECTORIZER.transform([model_text])
    x_char = CHAR_VECTORIZER.transform([model_text])
    x_features = hstack([x_word, x_char])

    predicted_aspects = []
    aspect_sentiments = {}

    for aspect in ASPECTS:
        pred_label = MODELS_SVM[aspect].predict(x_features)[0]
        if pred_label != 0:
            predicted_aspects.append(aspect)
            aspect_sentiments[aspect] = LABEL_TO_SENTIMENT.get(pred_label, "neutral")

    if not predicted_aspects:
        return {
            "review_id": 1,
            "aspects": ["none"],
            "aspect_sentiments": {"none": "neutral"},
        }

    return {
        "review_id": 1,
        "aspects": predicted_aspects,
        "aspect_sentiments": aspect_sentiments,
    }
