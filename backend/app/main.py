"""FastAPI application: JSON API under /api plus the built React app at /."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, engine, knnr
from .assets import Assets, load_assets
from .schemas import (
    CustomLocation, DistrictInfo, HealthResponse, HourlyOut, KnownLocation,
    LocationOut, MetaResponse, PredictRequest, PredictResponse, StateInfo,
    ValidationRequest, ValidationResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rainfall.api")

MAX_VALIDATION_DETAIL = 200  # per-day hyetographs sent to the browser


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Never crash on missing files: stay up and report the problem through
    # /api/health so the hosting platform's logs show a clear message.
    app.state.assets = None
    app.state.error = None
    try:
        app.state.assets = load_assets()
        knnr.warm_up()
    except Exception as exc:  # noqa: BLE001
        log.exception("Startup failed")
        app.state.error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="India Rainfall & Climate Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware, allow_origins=config.CORS_ORIGINS,
        allow_methods=["GET", "POST"], allow_headers=["*"],
    )


def get_assets(request: Request) -> Assets:
    a = request.app.state.assets
    if a is None:
        raise HTTPException(503, detail=request.app.state.error or "Models are not loaded yet.")
    return a


def _code_str(value) -> str:
    s = str(value)
    return s[:-2] if s.endswith(".0") else s


def _resolve(a: Assets, loc: KnownLocation | CustomLocation) -> dict:
    try:
        if isinstance(loc, KnownLocation):
            return engine.resolve_location(a, loc.state, loc.district)
        return engine.nearest_regime_location(
            a, loc.latitude, loc.longitude, loc.elevation, loc.distance_to_coast_km
        )
    except engine.LocationNotFound as exc:
        raise HTTPException(404, detail=str(exc)) from exc


def _location_out(loc: dict) -> LocationOut:
    return LocationOut(
        state_name=loc["state_name"], district_name=loc["district_name"],
        district_code=_code_str(loc["district_code"]),
        latitude=loc["latitude"], longitude=loc["longitude"], elevation=loc["elevation"],
        distance_to_coast_km=loc["distance_to_coast_km"],
        is_unlisted_location=loc["is_unlisted_location"],
        matched_from=loc.get("matched_from"),
    )


# ------------------------------------------------------------------- routes
@app.get("/api/health", response_model=HealthResponse)
def health(request: Request):
    a = request.app.state.assets
    if a is None:
        body = HealthResponse(status="error", detail=request.app.state.error)
        raise HTTPException(503, detail=body.model_dump())
    return HealthResponse(
        status="ok", districts=len(a.lookup), climate_models=len(a.climate_target_cols),
        knnr_windows=knnr.available_windows(),
    )


@app.get("/api/meta", response_model=MetaResponse)
def meta(a: Assets = Depends(get_assets)):
    states = []
    for name, grp in a.lookup.sort_values(["state_name", "district_name"]).groupby("state_name", sort=True):
        states.append(StateInfo(name=str(name), districts=[
            DistrictInfo(name=str(r.district_name), latitude=float(r.latitude), longitude=float(r.longitude))
            for r in grp.itertuples()
        ]))
    return MetaResponse(
        states=states, min_date=config.MIN_DATE, max_date=config.MAX_DATE,
        monsoon_months=[6, 7, 8, 9], knnr_windows=knnr.available_windows(),
        severity_order=config.SEVERITY_ORDER,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest, a: Assets = Depends(get_assets)):
    loc = _resolve(a, req.location)
    date_str = req.date.isoformat()
    try:
        pred = engine.predict(a, date_str, loc)
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction failed")
        raise HTTPException(500, detail=f"Prediction failed: {exc}") from exc

    hourly, status, message = None, "ok", None
    daily_total = pred["expected_precipitation_mm"]["weighted_mean"]

    if not req.knnr.enabled:
        status = "disabled"
    elif not knnr.library_available():
        status, message = "library_missing", "KNNR library not found; hourly disaggregation is unavailable."
    elif daily_total < 0.01:
        status, message = "no_rain", "Expected rainfall is effectively zero, so hourly disaggregation was skipped."
    else:
        try:
            res = knnr.ensemble(loc["district_code"], date_str, daily_total,
                                n_draws=req.knnr.draws, top_k=req.knnr.top_k)
            if res is None:
                status, message = "no_analogue", "No suitable historical KNNR analogue was found for this district/date."
            else:
                hourly = HourlyOut(**knnr.summarise_ensemble(res))
        except Exception as exc:  # noqa: BLE001
            # Rain/climate results stay valid even if the hourly part fails.
            log.exception("KNNR disaggregation failed")
            status, message = "error", f"Hourly disaggregation failed: {exc}"

    return PredictResponse(
        location=_location_out(loc), prediction=pred,
        hourly=hourly, hourly_status=status, hourly_message=message,
    )


@app.post("/api/validate", response_model=ValidationResponse)
def validate(req: ValidationRequest, a: Assets = Depends(get_assets)):
    if req.window not in knnr.available_windows():
        raise HTTPException(404, detail=f"KNNR {req.window}-day library not found.")
    code = None
    if req.scope == "district":
        if req.location is None:
            raise HTTPException(422, detail="location is required when scope='district'.")
        code = _resolve(a, req.location)["district_code"]
    out = knnr.run_validation(req.window, code, req.n_days, req.draws, req.top_k)
    if out is None:
        raise HTTPException(404, detail="No valid historical wet days with complete hourly data for this selection.")
    out["detail"] = out["detail"][:MAX_VALIDATION_DETAIL]
    return out


# ------------------------------------------------------- static React build
_dist = config.FRONTEND_DIST
if (_dist / "index.html").exists():
    if (_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, detail="Not found")
        candidate = (_dist / path).resolve()
        if path and candidate.is_file() and _dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "rainfall-api", "docs": "/docs", "note": "Frontend build not found."}
