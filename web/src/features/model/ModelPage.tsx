import { Link, useParams } from "react-router-dom";

import type { Gate, Interval, ModelDetail } from "../../shared/api/types";
import { duration, when } from "../../shared/lib/format";
import { STATUS } from "../../shared/theme/tokens";
import { Badge, EmptyState, PageHeader, Panel, QueryView, SkeletonBlock } from "../../shared/ui";
import { PathRail } from "../path/PathRail";
import { GateList } from "./GateList";
import { IntervalPlot } from "./IntervalPlot";
import styles from "./Model.module.css";
import { useModel } from "./api";

function marginsFrom(gates: Gate[]) {
  const overall = gates.find((g) => g.gate === "model.non_inferiority");
  const seg = gates.find((g) => g.gate === "model.segment_non_inferiority");
  return {
    margin: Number(overall?.evidence["margin"] ?? 0.02),
    marketMargin: Number(seg?.evidence["margin"] ?? 0.08),
    minQueries: Number(seg?.evidence["min_queries"] ?? 300),
  };
}

function Card({ card }: { card: Record<string, unknown> }) {
  const limits = Array.isArray(card["limits"]) ? (card["limits"] as string[]) : [];
  const window = card["training_window"] as { train?: string[]; holdout?: string[] } | string[] | undefined;
  const days = Array.isArray(window) ? window : [...(window?.train ?? []), ...(window?.holdout ?? [])];
  return (
    <dl className={styles.card}>
      {typeof card["intended_use"] === "string" && (
        <div>
          <dt>Intended use</dt>
          <dd>{card["intended_use"]}</dd>
        </div>
      )}
      {days.length > 0 && (
        <div>
          <dt>Training window</dt>
          <dd>
            {days[0]} to {days[days.length - 1]}
          </dd>
        </div>
      )}
      {typeof card["note"] === "string" && (
        <div>
          <dt>Note</dt>
          <dd>{card["note"]}</dd>
        </div>
      )}
      {limits.length > 0 && (
        <div>
          <dt>Known limits</dt>
          <dd>
            <ul>
              {limits.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </dd>
        </div>
      )}
    </dl>
  );
}

function Body({ d }: { d: ModelDetail }) {
  const m = d.model;
  const status = STATUS[m.status] ?? { label: m.status, tone: "quiet" as const };
  const metrics = m.metrics ?? {};
  const { margin, marketMargin, minQueries } = marginsFrom(d.gates);
  return (
    <>
      <PageHeader
        title={m.version}
        lead={`${m.kind === "ranker" ? "Search ranker" : "Homepage two-tower"}, owned by ${m.owner}. Registered ${when(m.created_at)} by ${m.pipeline_run}.`}
      >
        <Badge tone={status.tone}>{status.label}</Badge>
      </PageHeader>
      <Panel className={styles.railPanel}>
        <PathRail path={d.path} version={m.version} />
      </Panel>
      <div className={styles.grid}>
        <div className={styles.main}>
          {metrics.overall && metrics.markets && (
            <Panel
              title="Offline, against production"
              aside={
                <span className={styles.aside}>
                  {metrics.metric}, {metrics.weighting ?? "unweighted"}, against {metrics.champion}
                </span>
              }
            >
              <IntervalPlot
                overall={metrics.overall as Interval}
                markets={metrics.markets}
                margin={margin}
                marketMargin={marketMargin}
                minQueries={minQueries}
              />
              {metrics.holdout_days && (
                <p className={styles.note}>
                  Holdout: {metrics.holdout_days.join(", ")}, after every day the production model trained on.
                </p>
              )}
            </Panel>
          )}
          <Panel title="Gates">
            {d.gates.length ? (
              <GateList gates={d.gates} />
            ) : (
              <EmptyState title="No gate has run on this version">
                It was adopted from the ad-hoc process, not trained by the pipeline.
              </EmptyState>
            )}
          </Panel>
        </div>
        <div className={styles.aside2}>
          <Panel title="Deployments">
            {d.deployments.length === 0 ? (
              <p className={styles.note}>Not deployed.</p>
            ) : (
              <ol className={styles.deploys}>
                {d.deployments.map((dep) => (
                  <li key={dep.id}>
                    <div className={styles.deployHead}>
                      <span>#{dep.id}</span>
                      <Badge
                        tone={
                          dep.state === "promoted"
                            ? "pass"
                            : dep.state === "failed" || dep.state === "rolled_back"
                              ? "stop"
                              : "live"
                        }
                      >
                        {dep.state.replace("_", " ")}
                      </Badge>
                    </div>
                    <span className={styles.note}>
                      {dep.trigger}, {when(dep.started_at)}
                      {dep.finished_at ? `, ${duration(dep.started_at, dep.finished_at)}` : ""}
                    </span>
                    <ol className={styles.steps}>
                      {dep.steps.map((s, i) => (
                        <li key={i} className={s.ok ? styles.stepOk : styles.stepBad}>
                          <strong>{s.step.replace(/_/g, " ")}</strong> {s.detail}
                        </li>
                      ))}
                    </ol>
                  </li>
                ))}
              </ol>
            )}
          </Panel>
          <Panel title="History">
            <ol className={styles.history}>
              {d.transitions.map((t) => (
                <li key={t.id}>
                  <span className={styles.histWhen}>{when(t.at)}</span>
                  <span>
                    <strong>{STATUS[t.to_status]?.label ?? t.to_status}</strong> by {t.actor}: {t.reason}
                  </span>
                </li>
              ))}
            </ol>
          </Panel>
          <Panel title="Model card">
            <Card card={m.model_card ?? {}} />
          </Panel>
          <Panel title="Lineage">
            <dl className={styles.card}>
              <div>
                <dt>Pipeline run</dt>
                <dd>
                  <Link to={`/runs/${m.pipeline_run}`}>{m.pipeline_run}</Link>
                </dd>
              </div>
              <div>
                <dt>Code</dt>
                <dd>{m.code_commit}</dd>
              </div>
              <div>
                <dt>Data snapshot</dt>
                <dd>{m.data_snapshot}</dd>
              </div>
              {Object.entries(m.artifacts ?? {}).map(([name, a]) => (
                <div key={name}>
                  <dt>{name.replace(/_/g, " ")}</dt>
                  <dd title={a.sha256}>sha256 {a.sha256.slice(0, 12)}</dd>
                </div>
              ))}
            </dl>
          </Panel>
          {d.standards && !d.standards.conforms && (
            <Panel title="Short of the standards">
              <ul className={styles.findings}>
                {d.standards.findings.map((f) => (
                  <li key={f.standard + f.detail}>
                    <strong>{f.standard.replace("_", " ")}</strong>: {f.detail}
                  </li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      </div>
    </>
  );
}

export function ModelPage() {
  const { version = "" } = useParams();
  const q = useModel(version);
  return (
    <QueryView query={q} skeleton={<SkeletonBlock label={`Loading ${version}`} rows={6} />}>
      {(d) => <Body d={d} />}
    </QueryView>
  );
}
