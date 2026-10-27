import type { Station, StationState } from "../theme/tokens";

export type PathStation = {
  station: Station;
  state: StationState;
  gates: string[];
  note?: string | null;
  waived?: boolean;
};

export type Offline = { metric: string; delta: number; low: number; high: number; champion: string };

export type ModelSummary = {
  version: string;
  kind: "ranker" | "tower";
  status: string;
  created_at: number;
  pipeline_run: string;
  owner: string;
  code_commit: string;
  data_snapshot: string;
  offline?: Offline;
  path?: PathStation[];
};

export type Route = {
  surface: "search" | "home";
  production: string;
  canary: string | null;
  share: number;
  canary_since: number | null;
  previous: string | null;
  updated_at: number;
};

export type Run = {
  run_id: string;
  kind: string;
  state: string;
  outcome: string | null;
  version: string | null;
  started_at: number;
  finished_at: number | null;
  engine?: string;
  error?: string | null;
  blocking: string[];
};

export type Deployment = {
  id: number;
  version: string;
  trigger: string;
  started_at: number;
  finished_at: number | null;
  state: string;
  steps: { step: string; ok: boolean; detail: string; at: number }[];
  error: string | null;
};

export type Overview = {
  routing: Route[];
  models: Record<string, number>;
  production: Partial<Record<"ranker" | "tower", ModelSummary>>;
  recent: ModelSummary[];
  runs: Run[];
  open_deployments: Deployment[];
  standards: { checked: number; production_conforming: number; production: number };
  monitors: { surface: string; version: string; at: number; healthy: boolean }[];
};

export type Interval = {
  delta: number;
  low: number;
  high: number;
  n: number;
  champion?: number;
  candidate?: number;
};

export type Gate = {
  id: number;
  version: string | null;
  pipeline_run: string | null;
  stage: string;
  gate: string;
  passed: boolean;
  severity: "block" | "warn";
  blocking: boolean;
  waived_by: string | null;
  summary: string;
  evidence: Record<string, unknown>;
  at: number;
};

export type Transition = {
  id: number;
  from_status: string | null;
  to_status: string;
  at: number;
  actor: string;
  reason: string;
};

export type StandardFinding = { standard: string; detail: string };
export type StandardResult = {
  version: string;
  kind: string;
  status: string;
  conforms: boolean;
  findings: StandardFinding[];
};

export type ModelDetail = {
  model: ModelSummary & {
    features: unknown;
    training: Record<string, unknown>;
    metrics: {
      metric?: string;
      overall?: Interval;
      markets?: Record<string, Interval>;
      champion?: string;
      holdout_days?: string[];
      weighting?: string;
    };
    model_card: Record<string, unknown>;
    artifacts: Record<string, { key: string; sha256: string }>;
  };
  path: PathStation[];
  gates: Gate[];
  transitions: Transition[];
  deployments: Deployment[];
  standards: StandardResult | null;
};

export type Standard = { id: string; rule: string; why?: string };

export type StandardsReport = { standards: Standard[]; models: StandardResult[] };

export type RoutingReport = {
  current: Route[];
  log: {
    id: number;
    surface: string;
    production: string;
    canary: string | null;
    share: number;
    at: number;
    reason: string;
  }[];
};

export type SeasonRelease = {
  version: string;
  truth: "beneficial" | "neutral" | "harmful" | "broken";
  overall_delta: number;
  worst_market: string | null;
  worst_market_delta: number;
  platform: string;
  failed_step: string | null;
  note: string;
  day: number | null;
};

export type Season = {
  path: string;
  scenario: string;
  summary: Record<string, unknown> & {
    releases: number;
    harmful: number;
    harmful_promoted: number;
    beneficial: number;
    beneficial_promoted: number;
    beneficial_blocked: string[];
    harmful_exposure_share: number;
    hours_candidate_to_production_median: number | null;
  };
  releases: SeasonRelease[];
  data_blocks: { run: string; day: number; gates: string[]; causes: string[]; justified: boolean }[];
};
