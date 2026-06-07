document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('navToggle');
  const links = document.getElementById('navLinks');
  if (toggle && links) {
    toggle.addEventListener('click', () => links.classList.toggle('open'));
  }
});

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || 'Request failed');
  }
  return response.json();
}

function sentimentClass(label) {
  return `sentiment-${label.toLowerCase()}`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function truncate(text, max = 80) {
  if (text.length <= max) return text;
  return text.slice(0, max) + '…';
}
