const EVALUATION_INFO = {
  model: 'TF-IDF + Linear SVM',
  absa_micro_f1: 0.5817,
  accuracy_note: 'Validation performance measured on DeepX_validation.xlsx',
  pipeline: [
    'Arabic preprocessing',
    'TF-IDF word and character features',
    'One Linear SVM model per aspect',
    'JSON prediction output',
  ],
};

export default function JsonOutput({ prediction }) {
  const downloadReport = () => {
    if (!prediction) return;

    const report = {
      prediction,
      evaluation: EVALUATION_INFO,
    };

    const json = JSON.stringify(report, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'absa_prediction_report.json';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Raw JSON Output</h2>
      </div>
      <div className="json-actions">
        <button type="button" className="download-btn" onClick={downloadReport} disabled={!prediction}>
          Download JSON Report
        </button>
      </div>
      <pre className="json-block">
        <code>{prediction ? JSON.stringify(prediction, null, 2) : '// JSON output appears here'}</code>
      </pre>
    </section>
  );
}
