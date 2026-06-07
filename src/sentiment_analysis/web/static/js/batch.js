document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('csvFile');
  const browseBtn = document.getElementById('browseBtn');
  const uploadZone = document.getElementById('uploadZone');
  const uploadBtn = document.getElementById('uploadBtn');
  const progress = document.getElementById('batchProgress');
  const result = document.getElementById('batchResult');
  const errorPanel = document.getElementById('batchError');

  browseBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    uploadBtn.disabled = !fileInput.files.length;
    if (fileInput.files.length) {
      uploadZone.querySelector('p').textContent = `Selected: ${fileInput.files[0].name}`;
    }
  });

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      uploadBtn.disabled = false;
      uploadZone.querySelector('p').textContent = `Selected: ${e.dataTransfer.files[0].name}`;
    }
  });

  uploadBtn.addEventListener('click', async () => {
    if (!fileInput.files.length) return;

    result.classList.add('hidden');
    errorPanel.classList.add('hidden');
    progress.classList.remove('hidden');
    uploadBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
      const data = await apiFetch('/api/predict/batch', {
        method: 'POST',
        body: formData,
      });

      document.getElementById('batchTotal').textContent = data.total;
      const downloadLink = document.getElementById('downloadLink');
      downloadLink.href = data.download_url;

      const tbody = document.querySelector('#batchTable tbody');
      tbody.innerHTML = data.results.map((row) => {
        const conf = row.confidence || 0;
        return `<tr>
          <td>${truncate(row.text || '', 60)}</td>
          <td><span class="sentiment-badge ${sentimentClass(row.sentiment)}" style="font-size:0.8rem;padding:0.25rem 0.75rem">${row.sentiment}</span></td>
          <td>${formatPercent(conf)}</td>
        </tr>`;
      }).join('');

      result.classList.remove('hidden');
    } catch (err) {
      errorPanel.textContent = err.message;
      errorPanel.classList.remove('hidden');
    } finally {
      progress.classList.add('hidden');
      uploadBtn.disabled = false;
    }
  });
});
