import { useEffect, useMemo, useRef, useState } from "react";
import { latLngBounds, type LatLngBoundsExpression } from "leaflet";
import {
  CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, ZoomControl, useMap, useMapEvents,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { Meta } from "../types";

type LL = [number, number];

const INDIA_VIEW: LatLngBoundsExpression = [[6, 68], [36.5, 97.5]];
const MAX_BOUNDS: LatLngBoundsExpression = [[-2, 55], [45, 105]];
const DISTRICT_ZOOM = 9;

export interface DistrictPoint { state: string; name: string; lat: number; lon: number }

export const allDistricts = (meta: Meta): DistrictPoint[] =>
  meta.states.flatMap((s) =>
    s.districts.map((d) => ({ state: s.name, name: d.name, lat: d.latitude, lon: d.longitude })));

type CameraTarget =
  | { kind: "india" }
  | { kind: "bounds"; points: LL[] }
  | { kind: "point"; point: LL; zoom: number };

function BaseTiles() {
  return (
    <TileLayer
      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    />
  );
}

/** Animates the map to whatever the parent asks for; `delay` debounces typing. */
function CameraController({ camera, delay = 0 }: { camera: CameraTarget; delay?: number }) {
  const map = useMap();
  const key = JSON.stringify(camera);
  const first = useRef(true);
  useEffect(() => {
    if (first.current && camera.kind === "india") { first.current = false; return; } // initial bounds already set
    first.current = false;
    const run = () => {
      if (camera.kind === "point") {
        map.flyTo(camera.point, camera.zoom, { duration: 0.9 });
      } else if (camera.kind === "bounds" && camera.points.length > 1) {
        map.flyToBounds(latLngBounds(camera.points), { padding: [48, 48], maxZoom: 8, duration: 0.9 });
      } else if (camera.kind === "bounds" && camera.points.length === 1) {
        map.flyTo(camera.points[0], 8, { duration: 0.9 });
      } else {
        map.flyToBounds(INDIA_VIEW, { duration: 0.9 });;
      }
    };
    const t = window.setTimeout(run, delay);
    return () => window.clearTimeout(t);
  }, [key, delay, map]);
  return null;
}

function ClickToPick({ enabled, onPick }: { enabled: boolean; onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      if (enabled) onPick(Number(e.latlng.lat.toFixed(4)), Number(e.latlng.lng.toFixed(4)));
    },
  });
  return null;
}

/* ------------------------------------------------------------------ home */
interface HomeMapProps { districts: DistrictPoint[]; onPick: (state: string, district: string) => void }

export function HomeMap({ districts, onPick }: HomeMapProps) {
  return (
    <MapContainer bounds={INDIA_VIEW} maxBounds={MAX_BOUNDS} minZoom={4} maxZoom={12}
      preferCanvas zoomControl={false} scrollWheelZoom className="home-map">
      <BaseTiles />
      {districts.map((d) => (
        <CircleMarker key={`${d.state}|${d.name}`} center={[d.lat, d.lon]} radius={3.5}
          pathOptions={{ color: "#ffffff", weight: 1, fillColor: "#0e7c86", fillOpacity: 0.85 }}
          eventHandlers={{ click: () => onPick(d.state, d.name) }}>
          <Tooltip direction="top" offset={[0, -4]}>{d.name}, {d.state}: click to predict</Tooltip>
        </CircleMarker>
      ))}
      <ZoomControl position="bottomright" />
    </MapContainer>
  );
}

/* --------------------------------------------------------------- predict */
interface PredictMapProps {
  meta: Meta;
  mode: "known" | "custom";
  state: string;
  district: string;
  custom: { lat: number; lon: number } | null;      // null while the typed coordinates are invalid
  matched: { state: string; name: string } | null;  // known district a custom point borrowed its statistics from
  onPickDistrict: (state: string, district: string) => void;
  onPickPoint: (lat: number, lon: number) => void;
}

export default function MapView({ meta, mode, state, district, custom, matched, onPickDistrict, onPickPoint }: PredictMapProps) {
  const all = useMemo(() => allDistricts(meta), [meta]);
  const inState = useMemo(() => all.filter((d) => d.state === state), [all, state]);
  const outside = useMemo(() => (mode === "known" ? all.filter((d) => d.state !== state) : all), [all, state, mode]);
  const selected = inState.find((d) => d.name === district) ?? null;
  const matchedPoint = matched
    ? all.find((d) => d.state.toUpperCase() === matched.state.toUpperCase()
        && d.name.toUpperCase() === matched.name.toUpperCase()) ?? null
    : null;

  const [camera, setCamera] = useState<CameraTarget>({ kind: "india" });
  const lastState = useRef("");
  const selLat = selected?.lat, selLon = selected?.lon;
  const cLat = custom?.lat, cLon = custom?.lon;

  useEffect(() => {
    if (mode === "known") {
      if (state !== lastState.current) {
        // new state: show all of its districts first
        lastState.current = state;
        setCamera(inState.length
          ? { kind: "bounds", points: inState.map((d): LL => [d.lat, d.lon]) }
          : { kind: "india" });
      } else if (selLat !== undefined && selLon !== undefined) {
        // same state, new district: zoom in on it
        setCamera({ kind: "point", point: [selLat, selLon], zoom: DISTRICT_ZOOM });
      }
    } else if (cLat !== undefined && cLon !== undefined) {
      setCamera({ kind: "point", point: [cLat, cLon], zoom: DISTRICT_ZOOM });
    }
  }, [mode, state, selLat, selLon, cLat, cLon, inState]);

  return (
    <div className="mapbox tall" role="region" aria-label="Map of the selected location">
      <MapContainer bounds={INDIA_VIEW} maxBounds={MAX_BOUNDS} minZoom={4} maxZoom={12} scrollWheelZoom={false}>
        <BaseTiles />

        {outside.map((d) => (
          <CircleMarker key={`o|${d.state}|${d.name}`} center={[d.lat, d.lon]} radius={2.5}
            pathOptions={{ color: "transparent", weight: 0, fillColor: "#566772", fillOpacity: 0.35, interactive: false }} />
        ))}

        {mode === "known" && inState.map((d) => (
          <CircleMarker key={`s|${d.name}`} center={[d.lat, d.lon]} radius={5.5}
            pathOptions={{ color: "#ffffff", weight: 1.5, fillColor: "#0e7c86", fillOpacity: 0.95 }}
            eventHandlers={{ click: () => onPickDistrict(d.state, d.name) }}>
            <Tooltip direction="top" offset={[0, -6]}>{d.name}</Tooltip>
          </CircleMarker>
        ))}

        {mode === "known" && selected && (
          <>
            <CircleMarker center={[selected.lat, selected.lon]} radius={17} interactive={false}
              pathOptions={{ color: "#a93226", weight: 3, fillOpacity: 0, className: "halo-pulse" }} />
            <CircleMarker center={[selected.lat, selected.lon]} radius={8}
              pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#a93226", fillOpacity: 1 }}>
              <Tooltip permanent direction="top" offset={[0, -10]}>{selected.name}, {selected.state}</Tooltip>
            </CircleMarker>
          </>
        )}

        {mode === "custom" && custom && (
          <>
            {matchedPoint && (
              <>
                <Polyline positions={[[custom.lat, custom.lon], [matchedPoint.lat, matchedPoint.lon]]}
                  pathOptions={{ color: "#c98a12", weight: 2, dashArray: "6 6" }} />
                <CircleMarker center={[matchedPoint.lat, matchedPoint.lon]} radius={8}
                  pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#c98a12", fillOpacity: 1 }}>
                  <Tooltip permanent direction="top" offset={[0, -10]}>Matched: {matchedPoint.name}</Tooltip>
                </CircleMarker>
              </>
            )}
            <CircleMarker center={[custom.lat, custom.lon]} radius={17} interactive={false}
              pathOptions={{ color: "#a93226", weight: 3, fillOpacity: 0, className: "halo-pulse" }} />
            <CircleMarker center={[custom.lat, custom.lon]} radius={8}
              pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#a93226", fillOpacity: 1 }}>
              <Tooltip permanent direction="top" offset={[0, -10]}>Your point</Tooltip>
            </CircleMarker>
          </>
        )}

        <ClickToPick enabled={mode === "custom"} onPick={onPickPoint} />
        <CameraController camera={camera} delay={mode === "custom" ? 600 : 0} />
      </MapContainer>
      {mode === "custom" && <div className="map-hint">Click the map to place your point</div>}
    </div>
  );
}
