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
