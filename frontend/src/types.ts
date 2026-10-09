// Mirrors backend/app/schemas.py
export interface DistrictInfo { name: string; latitude: number; longitude: number }
export interface StateInfo { name: string; districts: DistrictInfo[] }
export interface Meta {
  states: StateInfo[]; min_date: string; max_date: string;
  monsoon_months: number[]; knnr_windows: number[]; severity_order: string[];
}

export type LocationInput =
  | { mode: "known"; state: string; district: string }
  | { mode: "custom"; latitude: number; longitude: number; elevation: number; distance_to_coast_km: number };

export interface KnnrOptions { enabled: boolean; draws: number; top_k: number }
export interface PredictRequest { date: string; location: LocationInput; knnr: KnnrOptions }

export interface LocationOut {
  state_name: string; district_name: string; district_code: string;
  latitude: number; longitude: number; elevation: number; distance_to_coast_km: number;
  is_unlisted_location: boolean;
  matched_from: { district_name: string; state_name: string; similarity_distance: number } | null;
}
export interface Prediction {
  date: string; rain_probability: number;
  severity_probabilities: { key: string; probability: number }[];
  expected_precipitation_mm: {
    weighted_mean: number; weighted_median: number;
    most_likely: { bin: string; probability: number; p25: number; p75: number };
  };
  climate: { key: string; label: string; unit: string; value: number; p10?: number | null; p90?: number | null }[];
}
export interface HourPoint { hour: number; mean: number; p10: number; p25: number; p75: number; p90: number }
export interface Hourly {
  hours: HourPoint[]; daily_total: number; peak_hour: number; peak_value: number; simulations: number;
  windows: number[]; any_pattern_matching: boolean; target_in_library: boolean;
  match_info: { window: number; matched_on: string }[]; analogue_dates: string[];
}
export type HourlyStatus = "ok" | "disabled" | "library_missing" | "no_rain" | "no_analogue" | "error";
export interface PredictResponse {
  location: LocationOut; prediction: Prediction;
  hourly: Hourly | null; hourly_status: HourlyStatus; hourly_message: string | null;
}

export interface ValidationRequest {
  window: 3 | 5 | 7; scope: "district" | "all";
  location?: { mode: "known"; state: string; district: string };
  n_days: number; draws: number; top_k: number;
}
export interface Metrics {
  mean_mae: number; median_mae: number; peak_hit_rate: number; mean_nse: number | null; median_nse: number | null;
}
export interface ValidationResponse {
  window: number;
  summary: { n_days: number; knnr: Metrics; naive: Metrics; beats_naive: boolean; match_counts: Record<string, number> };
  rows: { date: string; daily_total_mm: number; mae: number; peak_hour_hit: boolean; nse: number | null;
          naive_mae: number; naive_peak_hour_hit: boolean; naive_nse: number | null; match_type: string }[];
  detail: { date: string; daily_total_mm: number; actual: number[]; predicted: number[]; p10: number[]; p90: number[] }[];
}
