import re
import ast
import pandas as pd

# Arabic diacritics (tashkeel)
_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

# Keep Arabic, English, numbers, spaces, and common emoji ranges
_ALLOWED_CHARS_RE = re.compile(
    r"[^\u0600-\u06FFa-zA-Z0-9\s"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA70-\U0001FAFF"
    r"]+",
    flags=re.UNICODE,
)

_REPEATED_CHARS_RE = re.compile(r"(.)\1+")
_SPACES_RE = re.compile(r"\s+")


def preprocess_arabic_text(text: str) -> str:
    """Simple Arabic text preprocessing.

    Steps:
    1) remove diacritics (tashkeel)
    2) normalize Arabic letters
    3) remove repeated characters (e.g. حلوووو -> حلو)
    4) keep Arabic/English/numbers and emojis
    5) remove extra spaces
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = _DIACRITICS_RE.sub("", text)

    # Normalize letters
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )

    text = _REPEATED_CHARS_RE.sub(r"\1", text)
    text = _ALLOWED_CHARS_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()

    return text


def add_clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to review_text and create clean_text."""
    df = df.copy()
    df["clean_text"] = df["review_text"].apply(preprocess_arabic_text)
    return df


ASPECTS = [
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
]

SENTIMENT_TO_LABEL = {
    "positive": 1,
    "negative": 2,
    "neutral": 3,
}


def safe_literal_parse(value, default):
    """Safely parse a Python-literal-like string.

    Returns `default` when parsing fails or when value type does not match `default` type.
    """
    expected_type = type(default)
    if isinstance(value, expected_type):
        return value
    if pd.isna(value):
        return default
    if not isinstance(value, str):
        return default

    text = value.strip()
    if not text:
        return default

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return default

    if isinstance(parsed, expected_type):
        return parsed
    return default


def encode_aspect_labels(aspects_value, aspect_sentiments_value):
    """Create encoded labels for fixed aspects.

    Mapping:
    0=none, 1=positive, 2=negative, 3=neutral
    """
    aspects = safe_literal_parse(aspects_value, default=[])
    sentiments = safe_literal_parse(aspect_sentiments_value, default={})

    aspect_set = {str(a).strip().lower() for a in aspects}
    sentiment_map = {str(k).strip().lower(): str(v).strip().lower() for k, v in sentiments.items()}

    labels = {aspect: 0 for aspect in ASPECTS}

    for aspect in ASPECTS:
        if aspect in aspect_set:
            labels[aspect] = SENTIMENT_TO_LABEL.get(sentiment_map.get(aspect, ""), 0)

    return labels


def add_aspect_label_columns(
    df: pd.DataFrame,
    aspects_col: str = "aspects",
    sentiments_col: str = "aspect_sentiments",
) -> pd.DataFrame:
    """Parse aspect columns safely and append encoded label columns."""
    df = df.copy()
    encoded = df.apply(
        lambda row: encode_aspect_labels(row.get(aspects_col), row.get(sentiments_col)),
        axis=1,
    )
    labels_df = pd.DataFrame(encoded.tolist(), index=df.index)
    return pd.concat([df, labels_df], axis=1)


# Example usage:
# df = pd.read_csv("your_file.csv")
# df = add_clean_text_column(df)
# print(df[["review_text", "clean_text"]].head())
# df = add_aspect_label_columns(df)
# print(df[["food", "service", "price", "general"]].head())
