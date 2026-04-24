# Arabic Aspect-Based Sentiment Analysis (ABSA)

## Overview
This project implements an Arabic Aspect-Based Sentiment Analysis system that identifies aspects mentioned in a review and predicts the sentiment for each aspect (positive, negative, neutral). The system supports multiple aspects per review.

## Approach
A machine learning baseline is used:
- Arabic text preprocessing (normalization, removing diacritics, reducing repeated characters)
- TF-IDF features (word + character level)
- Logistic Regression (one model per aspect)

Each aspect is treated as a separate classification problem with labels:
- 0 = none
- 1 = positive
- 2 = negative
- 3 = neutral

## Aspects
food, service, price, cleanliness, delivery, ambiance, app_experience, general, none

## Evaluation
Model performance is evaluated using:
- Accuracy
- Classification report
- Micro F1 score
- ABSA Micro F1 (excluding none-only cases)

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Open and run:
arabic_absa.ipynb

3. Run all cells to generate:
submission.json

## Output Format
[
  {
    "review_id": 1,
    "aspects": ["food"],
    "aspect_sentiments": {
      "food": "positive"
    }
  }
]

If no aspect is detected:
{
  "review_id": 1,
  "aspects": ["none"],
  "aspect_sentiments": {
    "none": "neutral"
  }
}

## Files Included
- arabic_absa.ipynb
- arabic_preprocessing.py
- submission.json
- models/
- requirements.txt
- README.md

## Notes
- Dataset files are not included as required.
- The submission file is validated for format correctness.
- This is a reproducible baseline solution focused on clarity and correctness.