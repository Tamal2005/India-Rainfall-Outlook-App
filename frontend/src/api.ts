import type { Meta, PredictRequest, PredictResponse, ValidationRequest, ValidationResponse } from "./types";

// Same-origin by default (FastAPI serves the build). Set VITE_API_URL when the
// frontend is hosted separately, e.g. https://my-api.onrender.com
const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init, headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the server. Check your connection and try again.");
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      const d = body.detail;
      if (typeof d === "string") msg = d;
      else if (Array.isArray(d)) msg = d.map((e: { loc?: string[]; msg: string }) => `${(e.loc ?? []).slice(1).join(".")}: ${e.msg}`).join("; ");
      else if (d?.detail) msg = d.detail;
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export const getMeta = () => request<Meta>("/api/meta");
export const predict = (body: PredictRequest) =>
  request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify(body) });
export const validate = (body: ValidationRequest) =>
  request<ValidationResponse>("/api/validate", { method: "POST", body: JSON.stringify(body) });
