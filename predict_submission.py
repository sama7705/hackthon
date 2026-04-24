import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

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
        model_path = models_dir / f"{aspect}.joblib"
        if not model_path.exists():
            model_path = models_dir / f"{aspect}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find trained model for aspect '{aspect}'. "
                f"Expected: {models_dir / f'{aspect}.joblib'} or {models_dir / f'{aspect}.pkl'}"
            )
        models[aspect] = joblib.load(model_path)

    return models


def build_submission(unlabeled_df: pd.DataFrame, models: dict) -> list[dict]:
    records = []
    X_unlabeled = unlabeled_df[["combined_text"]]

    predicted_per_aspect = {
        aspect: models[aspect].predict(X_unlabeled).astype(int) for aspect in REQUIRED_ASPECTS
    }

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABSA submission JSON from trained models")
    parser.add_argument("--unlabeled", default="DeepX_unlabeled.xlsx", help="Unlabeled table")
    parser.add_argument("--models-dir", default="trained_models", help="Directory containing one model per aspect")
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
