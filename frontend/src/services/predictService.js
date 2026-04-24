const API_URL = 'http://127.0.0.1:8000/predict';

export async function predictReview(reviewText) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ review_text: reviewText }),
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    const detail = errorPayload?.detail || 'Prediction request failed';
    throw new Error(detail);
  }

  return response.json();
}
