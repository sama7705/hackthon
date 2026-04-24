const sentimentClass = {
  positive: 'badge badge--positive',
  negative: 'badge badge--negative',
  neutral: 'badge badge--neutral',
};

export default function ResultCards({ prediction }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Aspect Predictions</h2>
      </div>

      {!prediction ? (
        <p className="empty">شغّل التحليل لعرض الجوانب المكتشفة.</p>
      ) : (
        <div className="cards">
          {prediction.aspects.map((aspect) => {
            const sentiment = prediction.aspect_sentiments[aspect];
            return (
              <article key={aspect} className="result-card">
                <div>
                  <p className="label">Aspect</p>
                  <h3>{aspect}</h3>
                </div>
                <span className={sentimentClass[sentiment]}>{sentiment}</span>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
