document.addEventListener('DOMContentLoaded', async () => {
  const loading = document.getElementById('metricsLoading');
  const content = document.getElementById('metricsContent');
  const errorPanel = document.getElementById('metricsError');

  try {
    const data = await apiFetch('/api/metrics');
    loading.classList.add('hidden');
    content.classList.remove('hidden');

    document.getElementById('bestModelName').textContent = data.best_model || 'N/A';
    document.getElementById('bestF1').textContent = data.best_f1_macro
      ? data.best_f1_macro.toFixed(4) : 'N/A';
    document.getElementById('mlflowExp').textContent = data.mlflow_experiment || 'N/A';

    const models = data.models || [];
    const labels = models.map((m) => m.model_name.replace('tfidf_', '').replace('_', ' '));
    const f1Scores = models.map((m) => m.f1_macro || 0);
    const accuracies = models.map((m) => m.accuracy || 0);

    new Chart(document.getElementById('metricsChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'F1 Macro', data: f1Scores, backgroundColor: '#6366f1' },
          { label: 'Accuracy', data: accuracies, backgroundColor: '#22c55e' },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#8b92a8' } } },
        scales: {
          y: { beginAtZero: true, max: 1, ticks: { color: '#8b92a8' }, grid: { color: '#2d3348' } },
          x: { ticks: { color: '#8b92a8' }, grid: { display: false } },
        },
      },
    });

    const cardsEl = document.getElementById('modelCards');
    cardsEl.innerHTML = models.map((m) => `
      <div class="model-card">
        <h4>${m.model_name}</h4>
        <div class="model-stat"><span>Accuracy</span><span>${(m.accuracy || 0).toFixed(4)}</span></div>
        <div class="model-stat"><span>F1 Macro</span><span>${(m.f1_macro || 0).toFixed(4)}</span></div>
        <div class="model-stat"><span>F1 Weighted</span><span>${(m.f1_weighted || 0).toFixed(4)}</span></div>
        <div class="model-stat"><span>CV F1 Macro</span><span>${(m.cv_f1_macro_mean || 0).toFixed(4)}</span></div>
        ${m.confusion_matrix_url ? `<img src="${m.confusion_matrix_url}" alt="Confusion matrix" style="width:100%;margin-top:0.75rem;border-radius:8px">` : ''}
      </div>`).join('');
  } catch (err) {
    loading.classList.add('hidden');
    errorPanel.textContent = err.message;
    errorPanel.classList.remove('hidden');
  }
});
