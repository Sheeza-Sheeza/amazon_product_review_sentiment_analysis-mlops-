document.addEventListener('DOMContentLoaded', async () => {
  const statusEl = document.getElementById('healthStatus');

  try {
    const [health] = await Promise.all([
      apiFetch('/api/health'),
      loadSentimentDashboard(),
    ]);

    const modelStatus = health.model_loaded
      ? `<span class="status-ok"><span class="status-dot"></span>Model loaded (${health.model_source})</span>`
      : '<span class="status-warn"><span class="status-dot"></span>Model not loaded — train or configure MLFLOW_MODEL_URI</span>';
    statusEl.innerHTML = `
      <p>${modelStatus}</p>
      <p style="margin-top:0.5rem;color:var(--text-muted);font-size:0.9rem;">
        API v${health.version} · Status: ${health.status}
      </p>`;
  } catch (err) {
    if (statusEl) {
      statusEl.innerHTML = `<span class="error-panel">${err.message}</span>`;
    }
  }
});
