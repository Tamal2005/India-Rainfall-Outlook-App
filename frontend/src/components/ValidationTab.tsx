import { useState } from "react";
import { Area, Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { validate } from "../api";
import type { Meta, ValidationResponse } from "../types";
import { pct } from "../util";

interface Props { meta: Meta; mode: "known" | "custom"; state: string; district: string }

const f = (v: number | null, d = 3) => (v === null ? "–" : v.toFixed(d));

export default function ValidationTab({ meta, mode, state, district }: Props) {
  const [window, setWindow] = useState<3 | 5 | 7>((meta.knnr_windows[0] as 3 | 5 | 7) ?? 3);
  const [scope, setScope] = useState<"district" | "all">(mode === "known" ? "district" : "all");
  const [nDays, setNDays] = useState(200);
  const [draws, setDraws] = useState(20);
  const [topK, setTopK] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<ValidationResponse | null>(null);
  const [day, setDay] = useState<string>("");

  if (meta.knnr_windows.length === 0) {
    return <div className="notice warn">The KNNR library wasn't found on the server, so validation isn't available. Add the <code>disaggregation_knnr</code> folder and restart.</div>;
  }

  async function run() {
    setBusy(true); setError(null);
    try {
      const out = await validate({
        window, scope, n_days: nDays, draws, top_k: topK,
        location: scope === "district" ? { mode: "known", state, district } : undefined,
      });
      setRes(out); setDay(out.detail[0]?.date ?? "");
    } catch (e) {
      setRes(null); setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  }

  const chosen = res?.detail.find((d) => d.date === day);
  const chart = chosen?.actual.map((a, h) => ({ hour: h, observed: a, predicted: chosen.predicted[h], band: [chosen.p10[h], chosen.p90[h]] as [number, number] }));
  const s = res?.summary;
  const row = res?.rows.find((r) => r.date === day);

  return (
    <>
      <p className="hint" style={{ maxWidth: "75ch" }}>
        Leave-one-out backtest following Park &amp; Chung (2020). Each sampled wet day is removed from its own analogue pool, disaggregated,
        and compared with the hourly rainfall that actually fell, and against an even 24-hour spread.
      </p>
      <div className="panel" style={{ marginBottom: 18 }}>
        <div className="row2">
          <div className="field"><label htmlFor="v-win">Antecedent window</label>
            <select id="v-win" value={window} onChange={(e) => setWindow(Number(e.target.value) as 3 | 5 | 7)}>
              {meta.knnr_windows.map((w) => <option key={w} value={w}>{w}-day pattern</option>)}
            </select></div>
          <div className="field"><label htmlFor="v-scope">Scope</label>
            <select id="v-scope" value={scope} onChange={(e) => setScope(e.target.value as "district" | "all")}>
              <option value="district" disabled={mode !== "known"}>Selected district ({district || "none"})</option>
              <option value="all">All districts in library</option>
            </select></div>
        </div>
        <div className="row2">
          <div className="field"><label htmlFor="v-n">Days to sample: {nDays}</label>
            <input id="v-n" type="range" min={20} max={1000} step={20} value={nDays} onChange={(e) => setNDays(Number(e.target.value))} /></div>
          <div className="field"><label htmlFor="v-d">Simulations per day: {draws}</label>
            <input id="v-d" type="range" min={5} max={100} step={5} value={draws} onChange={(e) => setDraws(Number(e.target.value))} /></div>
        </div>
        <div className="field"><label htmlFor="v-k">Nearest neighbours (K): {topK}</label>
          <input id="v-k" type="range" min={3} max={50} value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></div>
        <button className="btn" style={{ width: "auto" }} onClick={run} disabled={busy}>{busy ? "Running backtest…" : "Run validation"}</button>
      </div>

      {error && <div className="notice error">{error}</div>}
      {!res && !error && !busy && <div className="empty">Pick a window and sample size, then run the validation. Larger samples take longer.</div>}

      {res && s && (
        <>
          <div className="tablewrap">
            <table>
              <thead><tr><th>Metric ({s.n_days} days)</th><th className="r">KNNR</th><th className="r">Even spread</th></tr></thead>
              <tbody>
                <tr><td>Mean hourly MAE (mm)</td><td className="r num">{f(s.knnr.mean_mae)}</td><td className="r num">{f(s.naive.mean_mae)}</td></tr>
                <tr><td>Median hourly MAE (mm)</td><td className="r num">{f(s.knnr.median_mae)}</td><td className="r num">{f(s.naive.median_mae)}</td></tr>
                <tr><td>Peak hour matched (±1 h)</td><td className="r num">{pct(s.knnr.peak_hit_rate)}</td><td className="r num">{pct(s.naive.peak_hit_rate)}</td></tr>
                <tr><td>Mean NSE</td><td className="r num">{f(s.knnr.mean_nse)}</td><td className="r num">{f(s.naive.mean_nse)}</td></tr>
                <tr><td>Median NSE</td><td className="r num">{f(s.knnr.median_nse)}</td><td className="r num">{f(s.naive.median_nse)}</td></tr>
              </tbody>
            </table>
          </div>
          <div className={`notice ${s.beats_naive ? "ok" : "error"}`} style={{ marginTop: 12 }}>
            {s.beats_naive
              ? "KNNR beats the even-spread baseline on mean MAE, so analogue matching is adding value."
              : "KNNR does not beat the even-spread baseline on mean MAE for this sample. Try a larger K, a different window, or check the size of the analogue pool."}
          </div>
          <p className="caption">Match types: {Object.entries(s.match_counts).map(([k, v]) => `${k} = ${v}`).join(", ")}</p>

          <section className="section">
            <h3>Observed vs disaggregated</h3>
            <div className="field" style={{ maxWidth: 280 }}>
              <label htmlFor="v-day">Validation day</label>
              <select id="v-day" value={day} onChange={(e) => setDay(e.target.value)}>
                {res.detail.map((d) => <option key={d.date} value={d.date}>{d.date}</option>)}
              </select>
            </div>
            {chosen && chart && (
              <>
                <p className="caption num" style={{ marginTop: 0 }}>Daily total {chosen.daily_total_mm.toFixed(1)} mm
                  {row && <> · MAE {row.mae.toFixed(3)} mm · peak hour {row.peak_hour_hit ? "matched" : "missed"} · NSE {f(row.nse)}</>}</p>
                <div className="chart" role="img" aria-label="Observed versus predicted hourly rainfall">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chart} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
                      <CartesianGrid stroke="var(--line)" vertical={false} />
                      <XAxis dataKey="hour" interval={1} tick={{ fill: "var(--muted)", fontSize: 12 }} height={30} />
                      <YAxis tick={{ fill: "var(--muted)", fontSize: 12 }} width={60} unit=" mm" />
                      <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 8 }}
                        formatter={(v: unknown) => (Array.isArray(v) ? `${(v[0] as number).toFixed(2)}–${(v[1] as number).toFixed(2)}` : (v as number).toFixed(2)) as string} />
                      <Area dataKey="band" name="KNNR P10–P90" stroke="none" fill="var(--sev-extreme)" fillOpacity={0.15} isAnimationActive={false} />
                      <Bar dataKey="observed" name="Observed" fill="var(--sev-moderate)" isAnimationActive={false} />
                      <Line dataKey="predicted" name="KNNR mean" stroke="var(--sev-extreme)" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </section>
        </>
      )}
    </>
  );
}
