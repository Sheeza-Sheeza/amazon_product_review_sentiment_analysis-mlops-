# ReviewSense

Production-ready NLP sentiment analysis for e-commerce reviews. Classifies Amazon product reviews as **positive**, **neutral**, or **negative** using a modular ML pipeline, a FastAPI web application, and MLOps tooling for experiment tracking, data versioning, and deployment.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-tracking%20%26%20registry-orange)
![DVC](https://img.shields.io/badge/DVC-data%20versioning-purple)

---

## Features

- **Sentiment classification** — Maps star ratings (1–2 negative, 3 neutral, 4–5 positive) and predicts sentiment from review text
- **Model comparison** — TF-IDF (Logistic Regression, Linear SVC, Naive Bayes) and sentence-embedding classifiers
- **Web dashboard** — Responsive UI with live positive/negative/neutral counts, single-review prediction, batch CSV upload, model metrics, and dataset analytics
- **MLflow integration** — Experiment tracking, artifact logging, and model registry support
- **DVC pipeline** — Reproducible data preparation and training stages
- **Docker + Render** — Containerized deployment on free-tier cloud hosting
- **CI/CD** — GitHub Actions for linting, testing, Docker builds, and deploy hooks

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML / NLP | scikit-learn, sentence-transformers, pandas |
| API / Web | FastAPI, Uvicorn, Jinja2, Chart.js |
| MLOps | MLflow, DVC, joblib |
| DevOps | uv, Docker, GitHub Actions, Render |

---

## Project Structure

```
.
├── config/
│   └── default.yaml              # Training & MLflow configuration
├── data/
│   ├── raw/                      # DVC-tracked raw dataset
│   └── processed/                # Prepared parquet (DVC output)
├── src/sentiment_analysis/
│   ├── api/                      # FastAPI app & routes
│   ├── data/                     # Loading & text preprocessing
│   ├── features/                 # TF-IDF & embedding vectorizers
│   ├── models/                   # Sklearn pipeline factory
│   ├── evaluation/               # Metrics & confusion matrices
│   ├── inference/                # Production predictor
│   ├── pipeline/                 # Training orchestration
│   ├── services/                 # Model, metrics, analytics services
│   ├── tracking/                 # MLflow utilities
│   └── web/                      # HTML templates, CSS, JavaScript
├── scripts/
│   ├── train.py                  # Training CLI
│   ├── predict.py                # Inference CLI
│   └── prepare_data.py           # DVC data prep stage
├── artifacts/                    # Models, metrics, reports
├── tests/                        # API & unit tests
├── eda.py                        # Exploratory data analysis script
├── dvc.yaml                      # DVC pipeline definition
├── Dockerfile
├── docker-compose.yml
├── render.yaml                   # Render deployment blueprint
└── pyproject.toml                # uv / Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone and install

```bash
git clone <your-repo-url>
cd Amazon_review_sentiment_analysis(mlops)

uv sync --all-groups
```

### 2. Pull dataset (DVC)

```bash
uv run dvc pull
```

If the dataset is not yet in DVC remote storage, copy it manually to `data/raw/7817_1.csv`.

### 3. Train a model

```bash
# Train all models (MLflow enabled)
uv run train-sentiment

# Fast training — single TF-IDF model, no MLflow
uv run train-sentiment --no-mlflow --models tfidf_logistic_regression
```

Artifacts are saved to `artifacts/`, including `artifacts/best_model/model.joblib`.

### 4. Start the web app

```bash
uv run serve
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Web Application

### Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Sentiment counts (positive / negative / neutral), system health |
| Predict | `/predict` | Classify a single review with confidence scores |
| Batch | `/batch` | Upload a CSV and download predictions |
| Metrics | `/metrics` | Model comparison charts and per-model reports |
| Analytics | `/analytics` | Dataset distributions, ratings, top brands |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/health` | Model load status & version |
| `POST` | `/api/predict` | Single review prediction (JSON) |
| `POST` | `/api/predict/batch` | Batch CSV upload |
| `GET` | `/api/metrics` | Training metrics JSON |
| `GET` | `/api/analytics` | Dataset analytics JSON |

**Single prediction example:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing and works perfectly!"}'
```

**Response:**

```json
{
  "text": "This product is amazing and works perfectly!",
  "sentiment": "positive",
  "confidence": 0.67,
  "probabilities": {
    "negative": 0.10,
    "neutral": 0.20,
    "positive": 0.67
  }
}
```

---

## Exploratory Data Analysis

Run the standalone EDA script before training:

```bash
uv run python eda.py
uv run python eda.py --data-path data/raw/7817_1.csv --output-dir .
```

Outputs `eda_report.txt` and `label_distribution.png`.

---

## ML Pipeline

### Label mapping

| Star Rating | Sentiment |
|-------------|-----------|
| 1 – 2 | negative |
| 3 | neutral |
| 4 – 5 | positive |

### Models compared

| Model | Feature type |
|-------|-------------|
| `tfidf_logistic_regression` | TF-IDF (1–2 grams) |
| `tfidf_linear_svc` | TF-IDF + calibrated SVM |
| `tfidf_multinomial_nb` | TF-IDF + Naive Bayes |
| `embedding_logistic_regression` | MiniLM embeddings + LR |

Best model is selected by **F1 macro** to handle class imbalance.

### Configuration

Edit `config/default.yaml` to change data paths, vectorizer settings, models, and MLflow options.

```yaml
mlflow:
  enabled: true
  experiment_name: amazon_review_sentiment
  register_best_model: true
  registered_model_name: amazon_review_sentiment_classifier
```

### CLI inference

```bash
uv run python scripts/predict.py --text "Terrible quality, broke after one day."
uv run python scripts/predict.py --input-file reviews.txt
```

---

## MLflow

### Local tracking

```bash
uv run train-sentiment
mlflow ui   # http://localhost:5000
```

### Model registry

Set environment variables to load models from the registry in production:

```bash
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_MODEL_URI=models:/amazon_review_sentiment_classifier/Production
```

The web app tries MLflow first, then falls back to `artifacts/best_model/model.joblib`.

---

## DVC Pipeline

```bash
# Run full pipeline: prepare data → train
uv run dvc repro

# Track or update dataset
uv run dvc add data/raw/7817_1.csv
uv run dvc push
```

Pipeline stages are defined in `dvc.yaml`:

1. **prepare_data** — Clean and export `data/processed/reviews.parquet`
2. **train** — Train models and write artifacts

---

## Docker

```bash
# Build and run app + MLflow server
docker compose up --build

# App:    http://localhost:8000
# MLflow: http://localhost:5000
```

Standalone build:

```bash
docker build -t reviewsense:latest .
docker run -p 8000:8000 reviewsense:latest
```

---

## Deploy to Render

1. Push the repository to GitHub.
2. Create a new **Web Service** on [Render](https://render.com) and connect the repo.
3. Render reads `render.yaml` automatically.
4. Set optional secrets in the Render dashboard:
   - `MLFLOW_TRACKING_URI`
   - `MLFLOW_MODEL_URI`
   - `RENDER_DEPLOY_HOOK` (for CI/CD auto-deploy)

Ensure `artifacts/best_model/model.joblib` is included in the Docker image or loaded via MLflow at runtime.

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server port |
| `MODEL_PATH` | `artifacts/best_model/model.joblib` | Local model artifact |
| `MLFLOW_MODEL_URI` | — | MLflow registry URI (overrides local path) |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking server |
| `DATA_PATH` | `data/raw/7817_1.csv` | Dataset for analytics |
| `ARTIFACTS_DIR` | `artifacts` | Training artifacts directory |

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request:

1. **Lint** — `ruff check`
2. **Test** — `pytest`
3. **Docker** — Build image and smoke-test `/health`
4. **Deploy** — Trigger Render deploy hook (when `RENDER_DEPLOY_HOOK` secret is set)

---

## Development

```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Add a dependency
uv add <package>
uv add --dev <package>
```

---

## Dataset

The project uses an Amazon product reviews dataset (`7817_1.csv`) with 1,597 rows and 27 columns. Key fields:

- `reviews.text` — Review body
- `reviews.rating` — Star rating (1–5)
- `brand`, `name`, `categories` — Product metadata

**Known data quality notes:**

- 420 rows have missing ratings (excluded from training)
- Heavy class imbalance toward positive reviews (~83%)
- Several metadata columns are fully or mostly null

---

## License

MIT (or specify your license here).
