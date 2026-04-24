export default function ReviewInput({ value, onChange, onAnalyze, sampleReviews }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>إدخال المراجعة</h2>
      </div>

      <textarea
        className="review-textarea"
        placeholder="اكتب مراجعة العميل هنا..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />

      <div className="sample-list">
        {sampleReviews.map((sample) => (
          <button key={sample} type="button" className="sample-btn" onClick={() => onChange(sample)}>
            {sample}
          </button>
        ))}
      </div>

      <button type="button" className="analyze-btn" onClick={onAnalyze}>
        Analyze
      </button>
    </section>
  );
}
