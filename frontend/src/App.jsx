import { useMemo, useState } from 'react';
import ReviewInput from './components/ReviewInput';
import ResultCards from './components/ResultCards';
import JsonOutput from './components/JsonOutput';
import StatsPanel from './components/StatsPanel';
import HowItWorks from './components/HowItWorks';

const SAMPLE_REVIEWS = [
  'الأكل ممتاز بس الخدمة بطيئة',
  'السعر غالي والتوصيل اتأخر',
  'التطبيق سهل والمكان نظيف',
];

const aspectRules = [
  { aspect: 'food', terms: ['أكل', 'اكل', 'طعام'] },
  { aspect: 'service', terms: ['خدمة', 'الخدمة'] },
  { aspect: 'price', terms: ['سعر', 'السعر', 'غالي', 'غلاء'] },
  { aspect: 'delivery', terms: ['توصيل', 'التوصيل', 'اتأخر', 'تأخر'] },
  { aspect: 'app_experience', terms: ['تطبيق', 'التطبيق', 'ابلكيشن'] },
  { aspect: 'cleanliness', terms: ['نظيف', 'نظافة', 'النظافة'] },
  { aspect: 'ambiance', terms: ['المكان', 'أجواء', 'ديكور'] },
  { aspect: 'general', terms: ['ممتاز', 'سيء', 'عادي'] },
];

const positiveTerms = ['ممتاز', 'رائع', 'جيد', 'سهل', 'نظيف', 'لذيذ'];
const negativeTerms = ['سيء', 'سيئة', 'بطيء', 'بطيئة', 'غالي', 'اتأخر', 'تأخر'];

function inferSentiment(review) {
  if (negativeTerms.some((term) => review.includes(term))) return 'negative';
  if (positiveTerms.some((term) => review.includes(term))) return 'positive';
  return 'neutral';
}

function mockPredict(reviewText) {
  const detected = [];

  aspectRules.forEach((rule) => {
    if (rule.terms.some((term) => reviewText.includes(term))) {
      detected.push(rule.aspect);
    }
  });

  const aspects = detected.length ? [...new Set(detected)] : ['none'];
  const aspectSentiments = {};

  aspects.forEach((aspect) => {
    aspectSentiments[aspect] = aspect === 'none' ? 'neutral' : inferSentiment(reviewText);
  });

  return {
    review_id: 1,
    aspects,
    aspect_sentiments: aspectSentiments,
  };
}

export default function App() {
  const [review, setReview] = useState('');
  const [prediction, setPrediction] = useState(null);

  const onAnalyze = () => {
    const cleaned = review.trim();
    if (!cleaned) return;
    setPrediction(mockPredict(cleaned));
  };

  const stats = useMemo(() => {
    if (!prediction) return null;

    return prediction.aspects.reduce(
      (acc, aspect) => {
        const sentiment = prediction.aspect_sentiments[aspect];
        acc.detectedAspects += aspect === 'none' ? 0 : 1;
        if (sentiment === 'positive') acc.positive += 1;
        if (sentiment === 'negative') acc.negative += 1;
        if (sentiment === 'neutral') acc.neutral += 1;
        return acc;
      },
      { detectedAspects: 0, positive: 0, negative: 0, neutral: 0 }
    );
  }, [prediction]);

  return (
    <div className="page" dir="rtl">
      <div className="bg-glow bg-glow--top" />
      <div className="bg-glow bg-glow--bottom" />

      <main className="dashboard">
        <header className="hero">
          <p className="pill">Hackathon-ready Frontend</p>
          <h1>Arabic ABSA Analyzer</h1>
          <p>
            تحليل مراجعات العملاء العربية لاستخراج الجوانب وتحديد المشاعر لكل جانب (Mock
            Prediction).
          </p>
        </header>

        <ReviewInput
          value={review}
          onChange={setReview}
          onAnalyze={onAnalyze}
          sampleReviews={SAMPLE_REVIEWS}
        />

        <section className="grid two-col">
          <ResultCards prediction={prediction} />
          <StatsPanel stats={stats} />
        </section>

        <JsonOutput prediction={prediction} />
        <HowItWorks />
      </main>
    </div>
  );
}
