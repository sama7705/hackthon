import argparse
import json
from pathlib import Path

import pandas as pd

from arabic_preprocessing import ASPECTS, add_aspect_label_columns, add_clean_text_column
from baseline_absa_lr import build_model

LABEL_TO_SENTIMENT = {
    1: "positive",
    2: "negative",
    3: "neutral",
}

ALLOWED_ASPECTS = {
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
ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def ensure_review_id(df: pd.DataFrame) -> pd.DataFrame:
    if "review_id" in df.columns:
        return df

    out = df.copy()
    out.insert(0, "review_id", range(1, len(out) + 1))
    return out


def train_models(train_df: pd.DataFrame):
    models = {}
    X_train = train_df[["clean_text"]]

    for aspect in ASPECTS:
        if aspect not in train_df.columns:
            raise ValueError(
                f"Missing training label column '{aspect}'. "
                "Make sure train data includes parsed aspect columns."
            )
        y_train = train_df[aspect].astype(int)
        model = build_model()
        model.fit(X_train, y_train)
        models[aspect] = model

    return models


def build_submission(unlabeled_df: pd.DataFrame, models: dict) -> list[dict]:
    records = []
    X_unlabeled = unlabeled_df[["clean_text"]]

    predicted_per_aspect = {
        aspect: models[aspect].predict(X_unlabeled).astype(int) for aspect in ASPECTS
    }

    for pos, (_, row) in enumerate(unlabeled_df.iterrows()):
        review_id = int(row["review_id"])
        aspects = []
        aspect_sentiments = {}

        for aspect in ASPECTS:
            label = int(predicted_per_aspect[aspect][pos])
            if label != 0:
                aspects.append(aspect)
                aspect_sentiments[aspect] = LABEL_TO_SENTIMENT[label]

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


def validate_submission(file_path, unlabeled_data) -> bool:
    with open(file_path, "r", encoding="utf-8") as f:
        submission = json.load(f)

    errors = []

    if not isinstance(submission, list):
        print("Validation errors:")
        print("- Submission root must be a list of prediction records.")
        return False

    if len(submission) != len(unlabeled_data):
        errors.append(
            f"Row count mismatch: submission has {len(submission)} rows, "
            f"but unlabeled_data has {len(unlabeled_data)} rows."
        )

    if isinstance(unlabeled_data, pd.DataFrame):
        expected_review_ids = set(unlabeled_data["review_id"].astype(int).tolist())
    else:
        expected_review_ids = {int(row["review_id"]) for row in unlabeled_data}

    seen_review_ids = set()

    for idx, row in enumerate(submission, start=1):
        if not isinstance(row, dict):
            errors.append(f"Row {idx}: each record must be an object/dict.")
            continue

        review_id = row.get("review_id")
        aspects = row.get("aspects")
        aspect_sentiments = row.get("aspect_sentiments")

        if review_id is None:
            errors.append(f"Row {idx}: missing 'review_id'.")
        else:
            try:
                review_id = int(review_id)
            except (TypeError, ValueError):
                errors.append(f"Row {idx}: review_id '{review_id}' is not an integer.")
            else:
                seen_review_ids.add(review_id)
                if review_id not in expected_review_ids:
                    errors.append(
                        f"Row {idx}: review_id {review_id} does not exist in unlabeled_data."
                    )

        if not isinstance(aspects, list):
            errors.append(f"Row {idx}: 'aspects' must be a list.")
            aspects = []

        invalid_aspects = [a for a in aspects if a not in ALLOWED_ASPECTS]
        if invalid_aspects:
            errors.append(
                f"Row {idx}: invalid aspects {invalid_aspects}. "
                f"Allowed: {sorted(ALLOWED_ASPECTS)}."
            )

        if not isinstance(aspect_sentiments, dict):
            errors.append(f"Row {idx}: 'aspect_sentiments' must be an object/dict.")
            aspect_sentiments = {}

        if set(aspect_sentiments.keys()) != set(aspects):
            errors.append(
                f"Row {idx}: aspect_sentiments keys {sorted(aspect_sentiments.keys())} "
                f"must exactly match aspects {sorted(aspects)}."
            )

        invalid_sentiments = {
            k: v for k, v in aspect_sentiments.items() if v not in ALLOWED_SENTIMENTS
        }
        if invalid_sentiments:
            errors.append(
                f"Row {idx}: invalid sentiments {invalid_sentiments}. "
                f"Allowed: {sorted(ALLOWED_SENTIMENTS)}."
            )

    missing_review_ids = sorted(expected_review_ids - seen_review_ids)
    if missing_review_ids:
        errors.append(
            f"Missing review_ids from submission: {missing_review_ids[:20]}"
            + (" ..." if len(missing_review_ids) > 20 else "")
        )

    if errors:
        print("Validation errors:")
        for err in errors:
            print(f"- {err}")
        return False

    print("Submission validation passed with no errors.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ABSA submission JSON")
    parser.add_argument("--train", default="DeepX_train.xlsx", help="Labeled training table")
    parser.add_argument(
        "--unlabeled", default="DeepX_unlabeled.xlsx", help="Unlabeled table to predict"
    )
    parser.add_argument("--output", default="submission.json", help="Output JSON path")
    args = parser.parse_args()

    train_df = load_table(Path(args.train))
    unlabeled_df = load_table(Path(args.unlabeled))

    train_df = add_clean_text_column(train_df)
    train_df = add_aspect_label_columns(train_df)

    unlabeled_df = ensure_review_id(unlabeled_df)
    unlabeled_df = add_clean_text_column(unlabeled_df)

    models = train_models(train_df)
    submission = build_submission(unlabeled_df, models)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
