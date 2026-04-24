const statsConfig = [
  { key: 'detectedAspects', title: 'Detected Aspects' },
  { key: 'positive', title: 'Positive' },
  { key: 'negative', title: 'Negative' },
  { key: 'neutral', title: 'Neutral' },
];

export default function StatsPanel({ stats }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Statistics</h2>
      </div>
      <div className="stats-grid">
        {statsConfig.map((item) => (
          <article key={item.key} className="stat-card">
            <p>{item.title}</p>
            <strong>{stats ? stats[item.key] : 0}</strong>
          </article>
        ))}
      </div>
    </section>
  );
}
