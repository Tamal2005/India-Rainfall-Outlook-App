import { useState } from "react";
import { predict } from "../api";
import ControlPanel, { type FormState } from "../components/ControlPanel";
import MapView from "../components/MapView";
import Results from "../components/Results";
import ValidationTab from "../components/ValidationTab";
import type { Meta, PredictResponse } from "../types";

interface Props {
  meta: Meta | null;
  form: FormState | null;
  setForm: (next: FormState) => void;
  bootError: string | null;
  onRetry: () => void;
}

export default function PredictPage({ meta, form, setForm, bootError, onRetry }: Props) {
  const [tab, setTab] = useState<"predict" | "validate">("predict");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!form) return;
    setLoading(true); setError(null);
    try {
      setResult(await predict({
        date: form.date, knnr: form.knnr,
        location: form.mode === "known"
          ? { mode: "known", state: form.state, district: form.district }
          : { mode: "custom", latitude: form.latitude, longitude: form.longitude, elevation: form.elevation, distance_to_coast_km: form.distance_to_coast_km },
      }));
    } catch (e) {
      setResult(null); setError(e instanceof Error ? e.message : String(e));
    } finally { setLoading(false); }
  }

  const custom = form && form.mode === "custom" && Number.isFinite(form.latitude) && Number.isFinite(form.longitude)
    ? { lat: form.latitude, lon: form.longitude } : null;
  const mf = result?.location.is_unlisted_location ? result.location.matched_from : null;
  const matched = form?.mode === "custom" && mf ? { state: mf.state_name, name: mf.district_name } : null;

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <a className="homelink" href="#/">← Home</a>
          <h1>Prediction</h1>
        </div>
        <p>Choose a state and district, or enter a point, and the map follows your selection.</p>
      </header>

      {bootError ? (
        <div className="notice error" role="alert">
          <b>The server isn't ready.</b> {bootError}
          <div style={{ marginTop: 8 }}><button className="btn secondary" style={{ width: "auto" }} onClick={onRetry}>Try again</button></div>
        </div>
      ) : !meta || !form ? (
        <div className="skeleton" aria-busy="true" aria-label="Loading" />
      ) : (
        <div className="layout">
          <aside className="rail">
            <ControlPanel meta={meta} value={form} onChange={setForm} onSubmit={() => { setTab("predict"); run(); }} loading={loading} />
          </aside>
          <main>
            <MapView
              meta={meta} mode={form.mode} state={form.state} district={form.district}
              custom={custom} matched={matched}
              onPickDistrict={(state, district) => setForm({ ...form, mode: "known", state, district })}
              onPickPoint={(latitude, longitude) => setForm({ ...form, latitude, longitude })}
            />
            <div className="tabs" role="tablist">
              <button role="tab" aria-selected={tab === "predict"} onClick={() => setTab("predict")}>Prediction</button>
              <button role="tab" aria-selected={tab === "validate"} onClick={() => setTab("validate")}>Hourly validation</button>
            </div>
            {tab === "predict" ? (
              <div role="tabpanel" aria-live="polite">
                {error && <div className="notice error" role="alert">{error}</div>}
                {loading && <div className="skeleton" aria-busy="true" />}
                {!loading && result && <Results data={result} />}
                {!loading && !result && !error && (
                  <div className="empty">Choose a place and a date, then select Predict.</div>
                )}
              </div>
            ) : (
              <div role="tabpanel">
                <ValidationTab meta={meta} mode={form.mode} state={form.state} district={form.district} />
              </div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}
