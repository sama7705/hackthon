import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack

from arabic_preprocessing import ASPECTS, add_clean_text_column
from baseline_absa_lr import add_domain_context_column

LABEL_TO_SENTIMENT = {
    1: "positive",
    2: "negative",
    3: "neutral",
}

REQUIRED_ASPECTS = [
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
]


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def ensure_review_id(df: pd.DataFrame) -> pd.DataFrame:
    if "review_id" in df.columns:
        out = df.copy()
        out["review_id"] = pd.to_numeric(out["review_id"], errors="raise").astype(int)
        if out["review_id"].duplicated().any():
            dupes = out.loc[out["review_id"].duplicated(), "review_id"].tolist()
            raise ValueError(f"Duplicate review_id values in input: {dupes[:10]}")
        return out

    out = df.copy()
    out.insert(0, "review_id", range(1, len(out) + 1))
    return out


def load_models(models_dir: Path) -> dict:
    models = {}

    missing_in_config = [aspect for aspect in REQUIRED_ASPECTS if aspect not in ASPECTS]
    if missing_in_config:
        raise ValueError(f"Required aspects missing from ASPECTS config: {missing_in_config}")

    for aspect in REQUIRED_ASPECTS:
        word_path = models_dir / f"{aspect}_word_tfidf.pkl"
        char_path = models_dir / f"{aspect}_char_tfidf.pkl"
        lr_path = models_dir / f"{aspect}_lr.pkl"

        missing_paths = [str(p) for p in [word_path, char_path, lr_path] if not p.exists()]
        if missing_paths:
            raise FileNotFoundError(
                f"Missing model artifacts for aspect '{aspect}': {missing_paths}. "
                "Train with baseline_absa_lr.py to generate these files."
            )

        models[aspect] = {
            "word_vectorizer": joblib.load(word_path),
            "char_vectorizer": joblib.load(char_path),
            "classifier": joblib.load(lr_path),
        }

    return models


def build_submission(unlabeled_df: pd.DataFrame, models: dict) -> list[dict]:
    records = []
    X_unlabeled = unlabeled_df["combined_text"].fillna("").astype(str)

    predicted_per_aspect = {}
    for aspect in REQUIRED_ASPECTS:
        model_bundle = models[aspect]
        word_features = model_bundle["word_vectorizer"].transform(X_unlabeled)
        char_features = model_bundle["char_vectorizer"].transform(X_unlabeled)
        all_features = hstack([word_features, char_features], format="csr")
        predicted_per_aspect[aspect] = model_bundle["classifier"].predict(all_features).astype(int)

    for pos, (_, row) in enumerate(unlabeled_df.iterrows()):
        review_id = int(row["review_id"])
        aspects = []
        aspect_sentiments = {}

        for aspect in REQUIRED_ASPECTS:
            label = int(predicted_per_aspect[aspect][pos])
            if label != 0:
                aspects.append(aspect)
                aspect_sentiments[aspect] = LABEL_TO_SENTIMENT.get(label, "neutral")

        if not aspects:
            aspects = ["none"]
            aspect_sentiments = {"none": "neutral"}

        records.append(
            {
                "review_id": review_id,
                "aspects": aspects,
                "aspect_sentiments": aspect_sentiments,
            }
        )

    return records


def assert_no_missing_review_ids(records: list[dict], source_df: pd.DataFrame) -> None:
    expected_ids = set(source_df["review_id"].astype(int).tolist())
    seen_ids = {int(r["review_id"]) for r in records}

    missing = sorted(expected_ids - seen_ids)
    if missing:
        raise ValueError(f"Missing review_id values in submission: {missing[:20]}")

    if len(records) != len(source_df):
        raise ValueError(
            f"Record count mismatch. submission={len(records)}, source={len(source_df)}"
        )


def validate_submission(file_path, unlabeled_df) -> None:
    allowed_aspects = {
        "food",
        "service",
        "price",
        "cleanliness",
        "delivery",
        "ambiance",
        "app_experience",
        "general",
        "none",
    }
    allowed_sentiments = {"positive", "negative", "neutral"}
    errors = []

    with open(file_path, "r", encoding="utf-8") as f:
        submission = json.load(f)

    if len(submission) != len(unlabeled_df):
        errors.append(
            f"Row count mismatch: submission has {len(submission)} rows, "
            f"unlabeled data has {len(unlabeled_df)} rows."
        )

    expected_review_ids = set(pd.to_numeric(unlabeled_df["review_id"], errors="coerce").dropna().astype(int))
    submission_review_ids = set()

    for i, row in enumerate(submission):
        row_prefix = f"Row {i + 1}"

        review_id = row.get("review_id")
        try:
            review_id = int(review_id)
        except (TypeError, ValueError):
            errors.append(f"{row_prefix}: review_id is invalid ({review_id}).")
            review_id = None

        if review_id is not None:
            submission_review_ids.add(review_id)

        aspects = row.get("aspects")
        if not isinstance(aspects, list):
            errors.append(f"{row_prefix}: aspects must be a list.")
            aspects = []

        aspect_sentiments = row.get("aspect_sentiments")
        if not isinstance(aspect_sentiments, dict):
            errors.append(f"{row_prefix}: aspect_sentiments must be a dictionary.")
            aspect_sentiments = {}

        if set(aspect_sentiments.keys()) != set(aspects):
            errors.append(
                f"{row_prefix}: keys in aspect_sentiments must exactly match aspects."
            )

        invalid_aspects = [a for a in aspects if a not in allowed_aspects]
        if invalid_aspects:
            errors.append(f"{row_prefix}: invalid aspects found: {invalid_aspects}.")

        invalid_sentiments = [
            sentiment for sentiment in aspect_sentiments.values() if sentiment not in allowed_sentiments
        ]
        if invalid_sentiments:
            errors.append(f"{row_prefix}: invalid sentiments found: {invalid_sentiments}.")

        if "none" in aspects and len(aspects) > 1:
            errors.append(f"{row_prefix}: 'none' cannot appear with other aspects.")

    missing_review_ids = sorted(expected_review_ids - submission_review_ids)
    if missing_review_ids:
        errors.append(
            f"Missing review_ids in submission (first 20): {missing_review_ids[:20]}."
        )

    extra_review_ids = sorted(submission_review_ids - expected_review_ids)
    if extra_review_ids:
        errors.append(
            f"Unknown review_ids in submission (first 20): {extra_review_ids[:20]}."
        )

    if errors:
        print("Submission validation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Submission file is valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABSA submission JSON from trained models")
    parser.add_argument("--unlabeled", default="DeepX_unlabeled.xlsx", help="Unlabeled table")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing fitted vectorizers and LR model per aspect",
    )
    parser.add_argument("--output", default="submission.json", help="Output JSON path")
    args = parser.parse_args()

    unlabeled_df = load_table(Path(args.unlabeled))
    unlabeled_df = ensure_review_id(unlabeled_df)
    unlabeled_df = add_clean_text_column(unlabeled_df)
    unlabeled_df = add_domain_context_column(unlabeled_df)

    models = load_models(Path(args.models_dir))
    submission_data = build_submission(unlabeled_df, models)
    assert_no_missing_review_ids(submission_data, unlabeled_df)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)

    print("submission.json saved successfully")


if __name__ == "__main__":
    main()
