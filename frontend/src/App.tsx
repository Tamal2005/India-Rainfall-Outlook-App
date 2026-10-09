import { useCallback, useEffect, useState } from "react";
import { ApiError, getMeta } from "./api";
import type { FormState } from "./components/ControlPanel";
import Home from "./pages/Home";
import PredictPage from "./pages/PredictPage";
import { useRoute } from "./route";
import type { Meta } from "./types";
import { today } from "./util";

export default function App() {
  const route = useRoute();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [form, setForm] = useState<FormState | null>(null);

  const boot = useCallback(() => {
    setBootError(null);
    getMeta().then((m) => {
      setMeta(m);
      const first = m.states[0];
      setForm({
        mode: "known", state: first?.name ?? "", district: first?.districts[0]?.name ?? "",
        latitude: 20, longitude: 78, elevation: 200, distance_to_coast_km: 100,
        date: today(), knnr: { enabled: m.knnr_windows.length > 0, draws: 50, top_k: 10 },
      });
    }).catch((e) => setBootError(e instanceof ApiError ? e.message : String(e)));
  }, []);
  useEffect(boot, [boot]);

  // a district clicked on the home map becomes the prediction page's selection
  const pickDistrict = useCallback((state: string, district: string) => {
    setForm((f) => (f ? { ...f, mode: "known", state, district } : f));
  }, []);

  return route === "home"
    ? <Home meta={meta} bootError={bootError} onPick={pickDistrict} />
    : <PredictPage meta={meta} form={form} setForm={setForm} bootError={bootError} onRetry={boot} />;
}
