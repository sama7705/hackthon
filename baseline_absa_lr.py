import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from arabic_preprocessing import ASPECTS, add_clean_text_column


LABEL_NAMES = {
    0: "none",
    1: "positive",
    2: "negative",
    3: "neutral",
}


def ensure_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Use existing clean_text if available, otherwise create it from review_text."""
    if "clean_text" in df.columns:
        out = df.copy()
        out["clean_text"] = out["clean_text"].fillna("").astype(str)
        return out

    if "review_text" not in df.columns:
        raise ValueError("DataFrame must contain either 'clean_text' or 'review_text'.")

    return add_clean_text_column(df)


def build_model() -> Pipeline:
    """TF-IDF (word+char) + Logistic Regression pipeline."""
    features = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2),
                "clean_text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2),
                "clean_text",
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=500,
        solver="saga",
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline([("features", features), ("clf", clf)])


def evaluate_aspect(train_df: pd.DataFrame, valid_df: pd.DataFrame, aspect: str) -> None:
    """Train one model per aspect and print metrics on validation data."""
    if aspect not in train_df.columns or aspect not in valid_df.columns:
        raise ValueError(f"Missing aspect label column: {aspect}")

    model = build_model()
    X_train = train_df[["clean_text"]]
    y_train = train_df[aspect].astype(int)

    X_valid = valid_df[["clean_text"]]
    y_valid = valid_df[aspect].astype(int)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    acc = accuracy_score(y_valid, y_pred)
    f1_macro = f1_score(y_valid, y_pred, average="macro", zero_division=0)

    print("=" * 80)
    print(f"Aspect: {aspect}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print("Classification report:")
    print(
        classification_report(
            y_valid,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=[LABEL_NAMES[i] for i in [0, 1, 2, 3]],
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Arabic ABSA with TF-IDF + Logistic Regression"
    )
    parser.add_argument("--train", required=True, help="Path to training file (.csv or .xlsx)")
    parser.add_argument(
        "--valid", required=True, help="Path to validation file (.csv or .xlsx)"
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)

    train_df = load_table(train_path)
    valid_df = load_table(valid_path)

    train_df = ensure_clean_text(train_df)
    valid_df = ensure_clean_text(valid_df)

    for aspect in ASPECTS:
        evaluate_aspect(train_df, valid_df, aspect)


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    main()
