document.addEventListener('DOMContentLoaded', async () => {
  const loading = document.getElementById('analyticsLoading');
  const content = document.getElementById('analyticsContent');
  const errorPanel = document.getElementById('analyticsError');

  const chartDefaults = {
    responsive: true,
    plugins: { legend: { labels: { color: '#8b92a8' } } },
  };

  try {
    const data = await loadSentimentDashboard({
      loadingId: 'analyticsLoading',
      contentId: 'analyticsContent',
      errorId: 'analyticsError',
    });

    document.getElementById('totalReviews').textContent = data.total_reviews;
    document.getElementById('avgLength').textContent = `${data.avg_text_length} chars`;
    document.getElementById('imbalanceRatio').textContent = `${data.class_imbalance_ratio}:1`;

    const sentimentColors = { positive: '#22c55e', neutral: '#eab308', negative: '#ef4444' };
    const sentLabels = Object.keys(data.sentiment_distribution);
    new Chart(document.getElementById('sentimentChart'), {
      type: 'doughnut',
      data: {
        labels: sentLabels,
        datasets: [{
          data: sentLabels.map((l) => data.sentiment_distribution[l]),
          backgroundColor: sentLabels.map((l) => sentimentColors[l] || '#6366f1'),
        }],
      },
      options: chartDefaults,
    });

    const ratingLabels = Object.keys(data.rating_distribution);
    new Chart(document.getElementById('ratingChart'), {
      type: 'bar',
      data: {
        labels: ratingLabels.map((l) => `${l} ★`),
        datasets: [{
          data: ratingLabels.map((l) => data.rating_distribution[l]),
          backgroundColor: '#6366f1',
        }],
      },
      options: {
        ...chartDefaults,
        plugins: { legend: { display: false } },
        scales: {
          y: { ticks: { color: '#8b92a8' }, grid: { color: '#2d3348' } },
          x: { ticks: { color: '#8b92a8' }, grid: { display: false } },
        },
      },
    });

    if (data.top_brands.length) {
      new Chart(document.getElementById('brandsChart'), {
        type: 'bar',
        data: {
          labels: data.top_brands.map((b) => b.brand),
          datasets: [{
            data: data.top_brands.map((b) => b.count),
            backgroundColor: '#818cf8',
          }],
        },
        options: {
          ...chartDefaults,
          indexAxis: 'y',
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#8b92a8' }, grid: { color: '#2d3348' } },
            y: { ticks: { color: '#8b92a8' }, grid: { display: false } },
          },
        },
      });
    }
  } catch (err) {
    loading.classList.add('hidden');
    errorPanel.textContent = err.message;
    errorPanel.classList.remove('hidden');
  }
});
