export const SEVERITY_LABEL: Record<string, string> = {
  no_rain: "No rain", light: "Light", medium: "Medium", moderate: "Moderate", heavy: "Heavy", extreme: "Extreme",
};
export const titleCase = (s: string) => SEVERITY_LABEL[s] ?? s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;
export const mm = (v: number, digits = 1) => `${v.toFixed(digits)} mm`;
export const today = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};
export const hourLabel = (h: number) => `${String(h).padStart(2, "0")}:00`;
