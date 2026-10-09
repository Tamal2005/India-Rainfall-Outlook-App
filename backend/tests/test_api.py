import numpy as np
import pytest

from app import knnr

KNOWN = {"mode": "known", "state": "Kerala", "district": "wayanad"}   # case-insensitive match


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["districts"] == 23
    assert body["climate_models"] == 6
    assert body["knnr_windows"] == [3, 5, 7]


def test_meta_lists_states_and_districts(client):
    body = client.get("/api/meta").json()
    names = [s["name"] for s in body["states"]]
    assert names == sorted(names) and "KERALA" in names
    kerala = next(s for s in body["states"] if s["name"] == "KERALA")
    assert {"name", "latitude", "longitude"} <= kerala["districts"][0].keys()


def test_predict_known_district(client):
    r = client.post("/api/predict", json={"date": "2024-07-15", "location": KNOWN})
    assert r.status_code == 200, r.text
    body = r.json()

    probs = {s["key"]: s["probability"] for s in body["prediction"]["severity_probabilities"]}
    assert set(probs) == {"no_rain", "light", "medium", "moderate", "heavy", "extreme"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0 <= body["prediction"]["rain_probability"] <= 1
    assert {c["key"] for c in body["prediction"]["climate"]} >= {"temperature", "surface_pressure"}
    for c in body["prediction"]["climate"]:        # quantile model present in the mock artifacts
        assert c["p10"] is not None and c["p10"] <= c["p90"]
    assert body["location"]["is_unlisted_location"] is False

    assert body["hourly_status"] == "ok", body["hourly_message"]
    h = body["hourly"]
    assert len(h["hours"]) == 24
    # hourly values must conserve the predicted daily total
    assert h["daily_total"] == pytest.approx(body["prediction"]["expected_precipitation_mm"]["weighted_mean"], rel=1e-6)
    assert h["simulations"] == 3 * 50


def test_predict_custom_location_borrows_regime(client):
    loc = {"mode": "custom", "latitude": 10.5, "longitude": 76.2, "elevation": 40, "distance_to_coast_km": 12}
    r = client.post("/api/predict", json={"date": "2024-01-10", "location": loc})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["location"]["is_unlisted_location"] is True
    assert body["location"]["matched_from"]["district_name"]


def test_predict_hourly_disabled(client):
    r = client.post("/api/predict", json={"date": "2024-07-15", "location": KNOWN, "knnr": {"enabled": False}})
    body = r.json()
    assert body["hourly"] is None and body["hourly_status"] == "disabled"


def test_unknown_district_gives_404_with_suggestions(client):
    r = client.post("/api/predict", json={"date": "2024-07-15",
                                          "location": {"mode": "known", "state": "Kerala", "district": "Wayanadd"}})
    assert r.status_code == 404
    assert "WAYANAD" in r.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"date": "1999-01-01", "location": KNOWN},                       # before range
    {"date": "2024-07-15", "location": {"mode": "custom", "latitude": 95, "longitude": 0,
                                        "elevation": 0, "distance_to_coast_km": 0}},
    {"date": "not-a-date", "location": KNOWN},
])
def test_bad_input_is_422(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_validation_endpoint(client):
    r = client.post("/api/validate", json={
        "window": 3, "scope": "district", "location": KNOWN, "n_days": 40, "draws": 5, "top_k": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["n_days"] > 0
    assert len(body["detail"][0]["actual"]) == 24
    assert isinstance(body["summary"]["beats_naive"], bool)


def test_validation_requires_location_for_district_scope(client):
    r = client.post("/api/validate", json={"window": 3, "scope": "district", "n_days": 40})
    assert r.status_code == 422


def test_unknown_api_path_is_404(client):
    assert client.get("/api/nope").status_code == 404


def test_rotation_is_orthogonal_and_total_is_conserved():
    assert np.allclose(knnr.ROTATION_24 @ knnr.ROTATION_24.T, np.eye(24), atol=1e-8)
    frag = knnr.ROTATION_24 @ np.random.default_rng(1).dirichlet(np.full(24, 0.3))
    out = knnr.inverse_rotation(frag, 12.5)
    assert out.sum() == pytest.approx(12.5) and (out >= 0).all()
