function renderSentimentDashboard(data) {
  const dist = data.sentiment_distribution || {};
  const positive = dist.positive || 0;
  const negative = dist.negative || 0;
  const neutral = dist.neutral || 0;
  const labeled = data.labeled_reviews || positive + negative + neutral;
  const total = data.total_reviews || labeled;

  const pct = (count) =>
    labeled > 0 ? `${((count / labeled) * 100).toFixed(1)}% of labeled` : '';

  const set = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };

  set('positiveCount', positive.toLocaleString());
  set('negativeCount', negative.toLocaleString());
  set('neutralCount', neutral.toLocaleString());
  set('totalLabeled', labeled.toLocaleString());
  set('labeledReviews', labeled.toLocaleString());

  set('positivePct', pct(positive));
  set('negativePct', pct(negative));
  set('neutralPct', pct(neutral));
  set('totalReviewsNote', `${total.toLocaleString()} total in dataset`);
}

async function loadSentimentDashboard(options = {}) {
  const {
    loadingId = 'dashboardLoading',
    contentId = 'dashboardContent',
    errorId = 'dashboardError',
  } = options;

  const loading = document.getElementById(loadingId);
  const content = document.getElementById(contentId);
  const errorPanel = document.getElementById(errorId);

  try {
    const data = await apiFetch('/api/analytics');
    if (loading) loading.classList.add('hidden');
    if (errorPanel) errorPanel.classList.add('hidden');
    if (content) content.classList.remove('hidden');
    renderSentimentDashboard(data);
    return data;
  } catch (err) {
    if (loading) loading.classList.add('hidden');
    if (errorPanel) {
      errorPanel.textContent = err.message;
      errorPanel.classList.remove('hidden');
    }
    throw err;
  }
}
