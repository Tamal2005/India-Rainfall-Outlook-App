import type { PredictResponse } from "../types";
import { hourLabel, mm, pct, titleCase } from "../util";
import HourlyChart from "./HourlyChart";

function downloadCsv(r: PredictResponse) {
  if (!r.hourly) return;
  const lines = ["hour,mean_mm,p10_mm,p25_mm,p75_mm,p90_mm",
    ...r.hourly.hours.map((h) => [hourLabel(h.hour), h.mean, h.p10, h.p25, h.p75, h.p90].map((v) => typeof v === "number" ? v.toFixed(3) : v).join(","))];
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `hourly_rainfall_${r.location.district_name.replace(/\W+/g, "_")}_${r.prediction.date}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function Results({ data }: { data: PredictResponse }) {
  const { location: loc, prediction: p, hourly } = data;
  const exp = p.expected_precipitation_mm;
  const top = exp.most_likely;
  const label = `${loc.district_name}${loc.is_unlisted_location ? "" : `, ${loc.state_name}`}`;

  return (
    <>
      {loc.is_unlisted_location && loc.matched_from && (
        <div className="notice info">
          This point isn't in the district list. Rainfall statistics come from the most similar known district,
          <b> {loc.matched_from.district_name}, {loc.matched_from.state_name}</b>, and hourly patterns are drawn from that district's history.
        </div>
      )}

      <div className="place">
        <h2>{label}</h2>
        <span className="num">{loc.latitude.toFixed(3)}°N, {loc.longitude.toFixed(3)}°E · {Math.round(loc.elevation)} m</span>
        <span className="num">{new Date(p.date + "T00:00:00").toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })}</span>
      </div>

      <div className="headline">
        <div>
          <div className="big num">{exp.weighted_mean.toFixed(1)}<small>mm expected</small></div>
          <p className="big-caption">
            Chance of rain {pct(p.rain_probability)}. Most likely: <b>{titleCase(top.bin)}</b> ({pct(top.probability)}),
            which is typically {top.p25.toFixed(1)}–{top.p75.toFixed(1)} mm in the training data.
          </p>
          <div className="facts">
            <div className="fact"><b className="num">{pct(p.rain_probability, 0)}</b><span>Rain probability</span></div>
            <div className="fact"><b className="num">{exp.weighted_median.toFixed(1)} mm</b><span>Typical (median)</span></div>
            <div className="fact"><b className="num">{titleCase(top.bin)}</b><span>Most likely class</span></div>
          </div>
        </div>

        <div className="ladder" role="list" aria-label="Rainfall severity probabilities">
          <h3>Severity probability</h3>
          {p.severity_probabilities.map((s) => (
            <div key={s.key} role="listitem" className={`rung${s.key === top.bin ? " top" : ""}`}>
              <span className="label">{titleCase(s.key)}</span>
              <span className="track" aria-hidden="true">
                <span className="fill" style={{ width: `${Math.min(100, s.probability * 100)}%`, background: `var(--sev-${s.key})` }} />
              </span>
              <span className="pct num">{pct(s.probability, s.probability < 0.01 ? 2 : 1)}</span>
            </div>
          ))}
        </div>
      </div>

      <section className="section" aria-labelledby="hourly-h">
        <h3 id="hourly-h">Hourly rainfall</h3>
        {data.hourly_status === "ok" && hourly ? (
          <>
            <HourlyChart hourly={hourly} />
            <div className="facts" style={{ marginTop: 12 }}>
              <div className="fact"><b className="num">{mm(hourly.daily_total, 2)}</b><span>Daily total</span></div>
              <div className="fact"><b className="num">{hourLabel(hourly.peak_hour)}</b><span>Peak hour</span></div>
              <div className="fact"><b className="num">{hourly.simulations}</b><span>Simulations</span></div>
            </div>

            {hourly.any_pattern_matching ? (
              <div className="notice ok" style={{ marginTop: 12 }}>
                Antecedent-pattern matching was used for at least one window.
                {" "}{hourly.match_info.map((m) => `${m.window}-day: ${m.matched_on}`).join("; ")}.
              </div>
            ) : (
              <div className="notice warn" style={{ marginTop: 12 }}>
                <b>Seasonal matching only.</b>{" "}
                {hourly.target_in_library
                  ? "The library has no pattern-label columns, so analogues were picked by day of year alone."
                  : `${p.date} isn't in the library, so there's no known preceding rainfall to match. Analogues come from the same district and time of year.`}
                {" "}The 3-, 5- and 7-day windows are equivalent here.
              </div>
            )}
            <p className="caption">
              Hourly values are resampled from historical days and rescaled so they add up to the predicted daily total.
              The bands show disagreement between analogues, not a calibrated confidence interval.
            </p>

            <div className="section">
              <button className="btn secondary" style={{ width: "auto" }} onClick={() => downloadCsv(data)}>Download hourly CSV</button>
            </div>
            <details>
              <summary>Hourly table</summary>
              <div className="tablewrap">
                <table>
                  <thead><tr><th>Hour</th><th className="r">Mean</th><th className="r">P10</th><th className="r">P25</th><th className="r">P75</th><th className="r">P90</th></tr></thead>
                  <tbody>
                    {hourly.hours.map((h) => (
                      <tr key={h.hour}><td className="num">{hourLabel(h.hour)}</td>
                        {[h.mean, h.p10, h.p25, h.p75, h.p90].map((v, i) => <td key={i} className="r num">{v.toFixed(3)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
            {hourly.analogue_dates.length > 0 && (
              <details>
                <summary>Historical analogue dates ({hourly.analogue_dates.length})</summary>
                <p className="caption num">{hourly.analogue_dates.join(", ")}</p>
              </details>
            )}
          </>
        ) : data.hourly_status === "disabled" ? (
          <div className="notice info">Hourly distribution is switched off.</div>
        ) : (
          <div className={`notice ${data.hourly_status === "error" ? "error" : "info"}`}>
            {data.hourly_message ?? "Hourly distribution is unavailable."}
            {data.hourly_status === "error" && " Rainfall and climate results above are still valid."}
          </div>
        )}
      </section>

      <section className="section" aria-labelledby="climate-h">
        <h3 id="climate-h">Climate estimates</h3>
        <div className="cgrid">
          {p.climate.map((c) => (
            <div className="cell" key={c.key}>
              <span>{c.label}</span>
              <b className="num">{c.value.toFixed(2)}</b><small>{c.unit}</small>
              {c.p10 != null && c.p90 != null && (
                <small className="num" style={{ display: "block", marginLeft: 0 }}>p10–p90: {c.p10.toFixed(1)} to {c.p90.toFixed(1)}</small>
              )}
            </div>
          ))}
        </div>
        <p className="caption">
          Climatology-style model: it uses the date's season and the location's historical rainfall character, not today's weather.
          Treat results as expected conditions, not an operational forecast.
        </p>
      </section>

      <details>
        <summary>Raw response</summary>
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </details>
    </>
  );
}
