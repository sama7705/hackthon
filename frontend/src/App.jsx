import { useMemo, useState } from 'react';
import ReviewInput from './components/ReviewInput';
import ResultCards from './components/ResultCards';
import JsonOutput from './components/JsonOutput';
import StatsPanel from './components/StatsPanel';
import HowItWorks from './components/HowItWorks';
import { predictReview } from './services/predictService';

const SAMPLE_REVIEWS = [
  'الأكل ممتاز بس الخدمة بطيئة',
  'السعر غالي والتوصيل اتأخر',
  'التطبيق سهل والمكان نظيف',
];

export default function App() {
  const [review, setReview] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  const onAnalyze = async () => {
    const cleaned = review.trim();
    if (!cleaned) return;

    setError('');
    try {
      const result = await predictReview(cleaned);
      setPrediction(result);
    } catch (err) {
      setPrediction(null);
      setError(err.message || 'Failed to get prediction from backend');
    }
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
          <p>تحليل مراجعات العملاء العربية لاستخراج الجوانب وتحديد المشاعر لكل جانب.</p>
          {error ? <p style={{ color: '#ff7f7f', marginTop: '0.8rem' }}>{error}</p> : null}
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
