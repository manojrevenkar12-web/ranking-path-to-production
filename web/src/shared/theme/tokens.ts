/** Words, tones and motion shared by every screen, so the console names things one way. */

export type Tone = "pass" | "stop" | "warn" | "live" | "quiet" | "thread";

export const MOTION = {
  sew: 0.9,
  move: 0.32,
  quick: 0.14,
  stagger: 0.04,
  ease: [0.2, 0.75, 0.25, 1] as [number, number, number, number],
} as const;

/** Live screens refresh this often (ms). */
export const REFRESH_MS = 15_000;

export type Station = "data" | "model" | "export" | "staging" | "canary" | "production";

export const STATIONS: { id: Station; label: string; means: string }[] = [
  {
    id: "data",
    label: "Data",
    means: "Freshness, volume, nulls, ranges, duplicates, click rates, provenance",
  },
  { id: "model", label: "Offline", means: "IPS-weighted DCG against production, overall and per market" },
  { id: "export", label: "Export", means: "ONNX parity with the native model, loads in the serving runtime" },
  { id: "staging", label: "Staging", means: "Score parity, skew and latency on replayed requests" },
  { id: "canary", label: "Canary", means: "Interleaving: which model members preferred" },
  { id: "production", label: "Production", means: "Serving all traffic, with the previous model loaded" },
];

export type StationState = "passed" | "stopped" | "active" | "waiting" | "pending" | "skipped";

export const STATION_STATE: Record<StationState, { label: string; tone: Tone }> = {
  passed: { label: "Passed", tone: "pass" },
  stopped: { label: "Stopped", tone: "stop" },
  active: { label: "Now", tone: "live" },
  waiting: { label: "Waiting", tone: "warn" },
  pending: { label: "Not yet", tone: "quiet" },
  skipped: { label: "Not reached", tone: "quiet" },
};

export const STATUS: Record<string, { label: string; tone: Tone }> = {
  rejected: { label: "Rejected", tone: "stop" },
  candidate: { label: "Candidate", tone: "thread" },
  staging: { label: "In staging", tone: "live" },
  canary: { label: "In canary", tone: "live" },
  production: { label: "Production", tone: "pass" },
  archived: { label: "Retired", tone: "quiet" },
  rolled_back: { label: "Rolled back", tone: "stop" },
  failed: { label: "Deploy failed", tone: "stop" },
};

export const OUTCOME: Record<string, { label: string; tone: Tone }> = {
  candidate: { label: "Candidate", tone: "pass" },
  blocked: { label: "Blocked", tone: "stop" },
  adopted: { label: "Adopted", tone: "quiet" },
  error: { label: "Error", tone: "stop" },
  embeddings_ready: { label: "Embeddings ready", tone: "pass" },
  embeddings_failed: { label: "Embeddings failed", tone: "stop" },
};

export const SURFACE: Record<string, { label: string; model: string }> = {
  search: { label: "Search", model: "Ranker" },
  home: { label: "Homepage", model: "Two-tower" },
};

/** What each gate checks, in the words of someone reading why a model stopped. */
export const GATE: Record<string, string> = {
  "data.freshness": "Newest logs are recent",
  "data.volume": "Daily volume per market",
  "data.null_rate": "No feature lost values",
  "data.range": "Features within contract ranges",
  "data.duplicates": "Duplicate impressions",
  "data.label_rate": "Click rate per market",
  "data.provenance": "Every feature was observable at serving",
  "data.propensity": "Position bias measured",
  "model.non_inferiority": "Not worse than production",
  "model.segment_non_inferiority": "Not worse in any market",
  "model.too_good_to_be_true": "Gain is plausible",
  "model.importance_concentration": "No single feature dominates",
  "export.onnx_parity": "ONNX matches the native model",
  "export.target_compat": "Loads in the serving runtime",
  "export.size": "Model size",
  "serving.score_parity": "Serving scores match offline",
  "serving.skew": "Serving computes features as training saw them",
  "serving.latency": "Latency on staging",
  "serving.embedding_coverage": "Embeddings cover the catalogue",
  "canary.guard": "Members' preference in the canary",
};

export const gateLabel = (id: string) => GATE[id] ?? id;
