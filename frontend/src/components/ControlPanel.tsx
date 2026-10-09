import { useMemo } from "react";
import type { KnnrOptions, Meta } from "../types";

export interface FormState {
  mode: "known" | "custom";
  state: string;
  district: string;
  latitude: number;
  longitude: number;
  elevation: number;
  distance_to_coast_km: number;
  date: string;
  knnr: KnnrOptions;
}

interface Props {
  meta: Meta;
  value: FormState;
  onChange: (next: FormState) => void;
  onSubmit: () => void;
  loading: boolean;
}

export default function ControlPanel({ meta, value, onChange, onSubmit, loading }: Props) {
  const set = (patch: Partial<FormState>) => onChange({ ...value, ...patch });
  const districts = useMemo(
    () => meta.states.find((s) => s.name === value.state)?.districts ?? [],
    [meta, value.state],
  );
  const num = (key: "latitude" | "longitude" | "elevation" | "distance_to_coast_km") => (e: React.ChangeEvent<HTMLInputElement>) =>
    set({ [key]: e.target.value === "" ? NaN : Number(e.target.value) } as Partial<FormState>);

  const customInvalid = value.mode === "custom" &&
    [value.latitude, value.longitude, value.elevation, value.distance_to_coast_km].some(Number.isNaN);

  return (
    <form className="panel" onSubmit={(e) => { e.preventDefault(); onSubmit(); }} aria-label="Prediction inputs">
      <div className="field">
        <span className="legend" id="loc-mode">Location</span>
        <div className="seg" role="group" aria-labelledby="loc-mode">
          <button type="button" aria-pressed={value.mode === "known"} onClick={() => set({ mode: "known" })}>Known district</button>
          <button type="button" aria-pressed={value.mode === "custom"} onClick={() => set({ mode: "custom" })}>Custom point</button>
        </div>
      </div>

      {value.mode === "known" ? (
        <>
          <div className="field">
            <label htmlFor="state">State / Union Territory</label>
            <select id="state" value={value.state}
              onChange={(e) => {
                const st = meta.states.find((s) => s.name === e.target.value);
                set({ state: e.target.value, district: st?.districts[0]?.name ?? "" });
              }}>
              {meta.states.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
            </select>
          </div>
          <div className="field">
            <label htmlFor="district">District</label>
            <select id="district" value={value.district} onChange={(e) => set({ district: e.target.value })}>
              {districts.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
            </select>
          </div>
        </>
      ) : (
        <>
          <p className="hint">
            For a place missing from the list: click the map or type coordinates. Rainfall statistics are borrowed from the most similar known district.
            Measure distance to the coast with the ruler tool in Google Maps.
          </p>
          <div className="row2">
            <div className="field"><label htmlFor="lat">Latitude</label>
              <input id="lat" type="number" step="0.0001" min={-90} max={90} value={Number.isNaN(value.latitude) ? "" : value.latitude} onChange={num("latitude")} /></div>
            <div className="field"><label htmlFor="lon">Longitude</label>
              <input id="lon" type="number" step="0.0001" min={-180} max={180} value={Number.isNaN(value.longitude) ? "" : value.longitude} onChange={num("longitude")} /></div>
          </div>
          <div className="row2">
            <div className="field"><label htmlFor="elev">Elevation (m)</label>
              <input id="elev" type="number" step="1" min={-500} max={9000} value={Number.isNaN(value.elevation) ? "" : value.elevation} onChange={num("elevation")} /></div>
            <div className="field"><label htmlFor="coast">Distance to coast (km)</label>
              <input id="coast" type="number" step="1" min={0} max={3000} value={Number.isNaN(value.distance_to_coast_km) ? "" : value.distance_to_coast_km} onChange={num("distance_to_coast_km")} /></div>
          </div>
        </>
      )}

      <div className="field">
        <label htmlFor="date">Date</label>
        <input id="date" type="date" required min={meta.min_date} max={meta.max_date} value={value.date}
          onChange={(e) => set({ date: e.target.value })} />
      </div>

      <hr className="divider" />

      <label className="check">
        <input type="checkbox" checked={value.knnr.enabled} disabled={meta.knnr_windows.length === 0}
          onChange={(e) => set({ knnr: { ...value.knnr, enabled: e.target.checked } })} />
        Include hourly distribution
      </label>
      {value.knnr.enabled && (
        <>
          <div className="field">
            <label htmlFor="draws">Simulations per window: {value.knnr.draws}</label>
            <input id="draws" type="range" min={10} max={200} step={10} value={value.knnr.draws}
              onChange={(e) => set({ knnr: { ...value.knnr, draws: Number(e.target.value) } })} />
          </div>
          <div className="field">
            <label htmlFor="topk">Nearest neighbours (K): {value.knnr.top_k}</label>
            <input id="topk" type="range" min={3} max={50} value={value.knnr.top_k}
              onChange={(e) => set({ knnr: { ...value.knnr, top_k: Number(e.target.value) } })} />
          </div>
        </>
      )}

      <button className="btn" type="submit" disabled={loading || !value.date || customInvalid}>
        {loading ? "Predicting…" : "Predict"}
      </button>
      <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>Monsoon is taken as June–September, matching the training features.</p>
    </form>
  );
}
