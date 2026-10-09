import { Area, Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Hourly } from "../types";
import { hourLabel } from "../util";

export default function HourlyChart({ hourly }: { hourly: Hourly }) {
  const data = hourly.hours.map((h) => ({ ...h, band: [h.p10, h.p90] as [number, number] }));
  return (
    <>
      <div className="legend-row">
        <span><i className="swatch" style={{ background: "var(--sev-medium)" }} />Ensemble mean</span>
        <span><i className="swatch" style={{ background: "var(--sev-no_rain)" }} />10th–90th percentile</span>
        <span><i className="swatch" style={{ background: "var(--sev-heavy)" }} />25th / 75th percentile</span>
      </div>
      <div className="chart" role="img" aria-label="Hourly rainfall distribution chart">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="hour" tickFormatter={(h) => String(h).padStart(2, "0")} interval={1}
              tick={{ fill: "var(--muted)", fontSize: 12 }} label={{ value: "Hour of day", position: "insideBottom", offset: -2, fill: "var(--muted)", fontSize: 12 }} height={36} />
            <YAxis tick={{ fill: "var(--muted)", fontSize: 12 }} width={60} unit=" mm" />
            <Tooltip
              formatter={(v: unknown) =>(Array.isArray(v) ? `${(v[0] as number).toFixed(2)}–${(v[1] as number).toFixed(2)} mm` : `${(v as number).toFixed(2)} mm`) as string}
              labelFormatter={(h) => hourLabel(Number(h))}
              contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 8 }} />
            <Area dataKey="band" name="P10–P90" stroke="none" fill="var(--sev-no_rain)" fillOpacity={0.55} isAnimationActive={false} />
            <Bar dataKey="mean" name="Mean" fill="var(--sev-medium)" radius={[2, 2, 0, 0]} isAnimationActive={false} />
            <Line dataKey="p25" name="P25" stroke="var(--sev-heavy)" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
            <Line dataKey="p75" name="P75" stroke="var(--sev-heavy)" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
