import { useMemo } from "react";
import { HomeMap, allDistricts } from "../components/MapView";
import { go } from "../route";
import type { Meta } from "../types";

interface Props {
  meta: Meta | null;
  bootError: string | null;
  onPick: (state: string, district: string) => void;
}

export default function Home({ meta, bootError, onPick }: Props) {
  const districts = useMemo(() => (meta ? allDistricts(meta) : []), [meta]);

  return (
    <div className="home">
      <HomeMap districts={districts} onPick={(s, d) => { onPick(s, d); go("predict"); }} />

      <div className="home-ui">
        <header className="float-card home-bar">
          <span className="brand"><i aria-hidden="true" />India rainfall outlook</span>
          <button className="btn" style={{ width: "auto" }} onClick={() => go("predict")}>Open prediction</button>
        </header>

        <section className="float-card hero" style={{ animationDelay: "0s" }}>
          <h1>Rainfall for any district in India, on any date.</h1>
          <p>
            Expected rainfall, odds of light to extreme rain, an hourly profile and climate estimates,
            for every district on the map.
          </p>
          <button className="btn cta" onClick={() => go("predict")}>Start prediction →</button>
          <div className="chips">
            {meta ? (
              <>
                <span>{districts.length} districts</span>
                <span>{meta.states.length} states &amp; UTs</span>
              </>
            ) : bootError ? (
              <span className="chip-warn">Server not reachable: start the backend to see districts</span>
            ) : (
              <span>Loading districts…</span>
            )}
          </div>
          <p className="hint" style={{ margin: "12px 0 0" }}>Tip: click any dot on the map to predict that district.</p>
        </section>

        <aside className="home-side">
          <section className="float-card" style={{ animationDelay: "-1.5s" }}>
            <h2>About the app</h2>
            <p>
              Pick a district, or enter latitude, longitude, elevation and distance to the coast for a place that
              isn't listed. You get the chance and severity of rain, the expected amount in millimetres,
              how it is likely to spread across the 24 hours, and expected temperature, humidity, wind and pressure.
            </p>
            <p className="note">Expected conditions from season and local rainfall character, not a live weather forecast.</p>
          </section>

          <section className="float-card" style={{ animationDelay: "-3s" }}>
            <h2>How the models work</h2>
            <ol className="steps">
              <li><b>Rain gate.</b> A neural network estimates whether it rains at all.</li>
              <li><b>Severity.</b> A second network splits rainy days into light, medium, moderate, heavy and extreme. Both are calibrated on held-out data.</li>
              <li><b>Climate.</b> A multi-task network estimates temperature, humidity, wind, pressure and more, with p10–p90 ranges where available.</li>
              <li><b>Hourly profile.</b> Historical 3-, 5- and 7-day rain patterns are resampled (KNN) and rescaled to the predicted daily total.</li>
            </ol>
          </section>
        </aside>
      </div>
    </div>
  );
}
