# India Rainfall & Climate Prediction — web app

A FastAPI backend that serves your trained models, and a React + TypeScript frontend, shipped as **one Docker image**. The API and the UI share an origin, so there is no CORS setup and no second service to deploy.

```
rainfall_webapp/
├── backend/
│   ├── app/
│   │   ├── config.py      paths + constants (every path is an env var)
│   │   ├── assets.py      loads models/lookup once at startup
│   │   ├── engine.py      location lookup, features, two-stage prediction
│   │   ├── knnr.py        hourly disaggregation + leave-one-out validation
│   │   ├── schemas.py     request/response contract
│   │   └── main.py        routes, health check, serves the built frontend
│   ├── scripts/
│   │   ├── rebuild_district_lookup.py   your one-time fix, no longer imports the app
│   │   └── make_mock_artifacts.py       tiny fake models for tests / trying the UI
│   ├── tests/             13 API tests that run against the mock artifacts
│   └── requirements.txt   model libraries pinned to your versions
├── frontend/              Vite + React + TypeScript
├── data/                  <- put your artifacts here
│   ├── prepped/district_lookup.parquet
│   ├── models/            feature_cols.pkl, stage1_rain_gate.json, climate_*.json, ...
│   └── disaggregation_knnr/   knnr_3day.parquet, knnr_5day.parquet, knnr_7day.parquet
├── Dockerfile  docker-compose.yml  render.yaml
```

`spatial_regime_features.csv` is **not** needed at runtime. It is only an input to `rebuild_district_lookup.py`.

## 1. Add your artifacts

Copy the folders you already have into `data/`: `prepped/`, `models/` and `disaggregation_knnr/` (same file names as before). If `district_lookup.parquet` doesn't yet contain the 12 regime columns, run once:

```bash
cd backend
pip install -r requirements.txt
python scripts/rebuild_district_lookup.py --regime-path /path/to/spatial_regime_features.csv
```

The server starts even if files are missing and tells you exactly which ones in `/api/health` and in the page itself, instead of crash-looping.

## 2. Run with Docker (recommended)

```bash
docker compose up --build
# open http://localhost:8000      API docs: http://localhost:8000/docs
```

## 3. Run for development (hot reload)

```bash
# terminal 1: API
cd backend && pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# terminal 2: UI (proxies /api to :8000)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Try it before your models are ready: `python backend/scripts/make_mock_artifacts.py --out data` creates fake artifacts in the same layout. Predictions from them are meaningless. Delete them before adding the real files.

## 4. Deploy

The image listens on `$PORT`, exposes `/api/health`, and runs as a non-root user.

| Platform | How |
|---|---|
| **Render** | New → Blueprint → select the repo (`render.yaml` is included). Use a paid instance if the KNNR library is large. |
| **Railway / Fly.io / Google Cloud Run / Azure** | Deploy from the Dockerfile. Set the health check path to `/api/health`. |
| **Hugging Face Spaces** | New Space → Docker. Add `app_port: 8000` to the Space's README front-matter. |
| **Any VPS** | `docker compose up -d --build` behind Nginx or Caddy. |

Memory is dominated by the KNNR parquet files, which are loaded once at startup. Plan for roughly 2–3× their size on disk.

If your artifacts exceed GitHub's 100 MB file limit, use Git LFS (`git lfs track "*.parquet" "*.pkl"`) or build the image in CI and keep `data/` out of the repo.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port to listen on |
| `DATA_ROOT` | `<project>/data` | Parent of `prepped/`, `models/`, `disaggregation_knnr/` |
| `DATA_DIR`, `MODEL_DIR`, `KNNR_DIR` | derived from `DATA_ROOT` | Override any single folder |
| `CORS_ORIGINS` | empty | Comma-separated origins, only if the UI is hosted separately |
| `SHOW_PROGRESS` | `1` (`0` in Docker) | tqdm progress bars for model/library loading and validation |

To host the UI separately (e.g. Vercel), build it with `VITE_API_URL=https://your-api` and set `CORS_ORIGINS` on the API.

## API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | 200 when models are loaded, 503 with the reason otherwise |
| `GET /api/meta` | States, districts with coordinates, date range, available KNNR windows |
| `POST /api/predict` | Rain probability, severity odds, expected mm, climate values, hourly ensemble |
| `POST /api/validate` | Leave-one-out KNNR backtest vs an even 24 h spread |

```bash
curl -X POST localhost:8000/api/predict -H 'content-type: application/json' -d '{
  "date": "2024-07-15",
  "location": {"mode": "known", "state": "ANDAMAN AND NICOBAR ISLANDS", "district": "Nicobars"},
  "knnr": {"enabled": true, "draws": 50, "top_k": 10}
}'
```

A failure in the hourly step never loses the rain and climate results; the response carries `hourly_status` and `hourly_message` instead.

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && pytest
```

## Notes carried over from the Streamlit version

- The monsoon flag is June–September, exactly as in training.
- The model is climatology-style: it uses the date's season and the location's historical regime, not current weather. The UI says so.
- Keep the pinned `scikit-learn`, `xgboost`, `numpy` and `pandas` versions. The `.pkl` calibrators are version-sensitive.
- Logic is a direct port. One thing to check on your real data: validation compares the stored `h0…h23` columns directly with the disaggregated output, as the Streamlit app did, while disaggregation treats those same columns as Gram-Schmidt-rotated fragments. If your library stores rotated values, "observed" bars in the validation chart can look negative; if it stores plain hourly shares, the rotation step needs revisiting.
