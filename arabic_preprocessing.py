import re
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


# Example usage:
# df = pd.read_csv("your_file.csv")
# df = add_clean_text_column(df)
# print(df[["review_text", "clean_text"]].head())
