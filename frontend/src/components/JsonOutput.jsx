export default function JsonOutput({ prediction }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Raw JSON Output</h2>
      </div>
      <pre className="json-block">
        <code>{prediction ? JSON.stringify(prediction, null, 2) : '// JSON output appears here'}</code>
      </pre>
    </section>
  );
}
