document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictForm');
  const resultPanel = document.getElementById('predictResult');
  const errorPanel = document.getElementById('predictError');
  const badge = document.getElementById('sentimentBadge');
  const confidence = document.getElementById('confidenceValue');
  const bars = document.getElementById('probabilityBars');
  const btn = document.getElementById('predictBtn');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultPanel.classList.add('hidden');
    errorPanel.classList.add('hidden');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    const text = document.getElementById('reviewText').value.trim();
    try {
      const data = await apiFetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      badge.textContent = data.sentiment;
      badge.className = `sentiment-badge ${sentimentClass(data.sentiment)}`;
      confidence.textContent = formatPercent(data.confidence);

      const colors = { positive: '#22c55e', neutral: '#eab308', negative: '#ef4444' };
      bars.innerHTML = Object.entries(data.probabilities)
        .map(([label, prob]) => `
          <div class="prob-row">
            <span>${label}</span>
            <div class="prob-bar-bg">
              <div class="prob-bar-fill" style="width:${prob * 100}%;background:${colors[label] || '#6366f1'}"></div>
            </div>
            <span>${formatPercent(prob)}</span>
          </div>`)
        .join('');

      resultPanel.classList.remove('hidden');
    } catch (err) {
      errorPanel.textContent = err.message;
      errorPanel.classList.remove('hidden');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Analyze Sentiment';
    }
  });
});
