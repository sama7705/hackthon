const steps = ['Arabic preprocessing', 'TF-IDF features', 'Aspect-wise models', 'JSON prediction'];

export default function HowItWorks() {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>How it works</h2>
      </div>
      <div className="steps">
        {steps.map((step, i) => (
          <div key={step} className="step">
            <span>{i + 1}</span>
            <p>{step}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
