import { Link } from "react-router-dom";

import type { ModelSummary, Overview, Route } from "../../shared/api/types";
import { duration, pct, share, versionOf, when } from "../../shared/lib/format";
import { OUTCOME, STATUS, SURFACE, gateLabel } from "../../shared/theme/tokens";
import { Badge, EmptyState, PageHeader, Panel, QueryView, SkeletonBlock, Stat } from "../../shared/ui";
import { PathRail } from "./PathRail";
import styles from "./Path.module.css";
import { useOverview } from "./api";

function SurfaceCard({
  surface,
  route,
  production,
}: {
  surface: string;
  route: Route | undefined;
  production?: ModelSummary;
}) {
  const meta = SURFACE[surface] ?? { label: surface, model: "" };
  return (
    <Panel
      title={`${meta.label}`}
      aside={<span className={styles.kind}>{meta.model}</span>}
      className={styles.surface}
    >
      {route ? (
        <dl className={styles.routing}>
          <div>
            <dt>Serving</dt>
            <dd>
              {route.production.startsWith("ranker_") || route.production.startsWith("t") ? (
                <Link to={`/models/${versionOf(route.production)}`}>{versionOf(route.production)}</Link>
              ) : (
                route.production
              )}
              {production?.offline && (
                <span className={styles.sub}>
                  {pct(production.offline.delta)} offline vs {production.offline.champion}
                </span>
              )}
            </dd>
          </div>
          <div>
            <dt>Canary</dt>
            <dd>
              {route.canary ? (
                <>
                  <Link to={`/models/${versionOf(route.canary)}`}>{versionOf(route.canary)}</Link>
                  <span className={styles.sub}>
                    {share(route.share)} of traffic, interleaved
                    {route.canary_since ? `, since ${when(route.canary_since)}` : ""}
                  </span>
                </>
              ) : (
                <span className={styles.none}>None running</span>
              )}
            </dd>
          </div>
          <div>
            <dt>Rollback to</dt>
            <dd>
              {route.previous ? (
                versionOf(route.previous)
              ) : (
                <span className={styles.none}>Nothing loaded</span>
              )}
            </dd>
          </div>
        </dl>
      ) : (
        <p className={styles.none}>No model serves this surface yet: it runs on the popularity profile.</p>
      )}
    </Panel>
  );
}

function VersionRow({ v, first }: { v: ModelSummary; first: boolean }) {
  const status = STATUS[v.status] ?? { label: v.status, tone: "quiet" as const };
  return (
    <li className={styles.version}>
      <div className={styles.versionHead}>
        <Link to={`/models/${v.version}`} className={styles.versionId}>
          {v.version}
        </Link>
        <Badge tone={status.tone}>{status.label}</Badge>
        <span className={styles.meta}>
          {v.kind === "ranker" ? "Search ranker" : "Homepage two-tower"}, {when(v.created_at)}
        </span>
        {v.offline && (
          <span
            className={styles.offline}
            title={`${v.offline.metric} against ${v.offline.champion}, 95% interval`}
          >
            {pct(v.offline.delta)}{" "}
            <span className={styles.interval}>
              ({pct(v.offline.low)} to {pct(v.offline.high)})
            </span>
          </span>
        )}
      </div>
      {v.path && <PathRail path={v.path} version={v.version} size={first ? "full" : "compact"} sew={first} />}
    </li>
  );
}

function Body({ data }: { data: Overview }) {
  const routes = Object.fromEntries(data.routing.map((r) => [r.surface, r]));
  const prodCount = data.standards.production;
  return (
    <div className={styles.layout}>
      <div className={styles.surfaces}>
        <SurfaceCard surface="search" route={routes["search"]} production={data.production.ranker} />
        <SurfaceCard surface="home" route={routes["home"]} production={data.production.tower} />
      </div>
      <Panel
        title="Versions on the path"
        aside={<Link to="/models">All versions</Link>}
        className={styles.versions}
      >
        {data.recent.length === 0 ? (
          <EmptyState title="No model versions yet">
            The first continuous-training run registers one.
          </EmptyState>
        ) : (
          <ul className={styles.versionList}>
            {data.recent.map((v, i) => (
              <VersionRow key={v.version} v={v} first={i === 0} />
            ))}
          </ul>
        )}
      </Panel>
      <aside className={styles.side}>
        <Panel title="Latest pipeline runs" aside={<Link to="/runs">All runs</Link>}>
          {data.runs.length === 0 ? (
            <p className={styles.none}>No runs yet.</p>
          ) : (
            <ul className={styles.runs}>
              {data.runs.slice(0, 8).map((r) => {
                const o = OUTCOME[r.outcome ?? ""] ?? { label: r.state, tone: "quiet" as const };
                return (
                  <li key={r.run_id}>
                    <div className={styles.runHead}>
                      <Link to={`/runs/${r.run_id}`}>{r.run_id}</Link>
                      <Badge tone={o.tone}>{o.label}</Badge>
                    </div>
                    <span className={styles.meta}>
                      {when(r.started_at)}
                      {r.finished_at && r.finished_at > r.started_at
                        ? `, took ${duration(r.started_at, r.finished_at)}`
                        : ""}
                    </span>
                    {r.blocking.length > 0 && (
                      <span className={styles.blocking}>{r.blocking.map(gateLabel).join(", ")}</span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </Panel>
        <Panel title="Lifecycle standards" aside={<Link to="/standards">Details</Link>}>
          <div className={styles.stats}>
            <Stat
              value={`${data.standards.production_conforming}/${prodCount}`}
              label="production models conform"
              tone={prodCount && data.standards.production_conforming === prodCount ? "pass" : "warn"}
            />
            <Stat
              value={data.open_deployments.length}
              label="deployments in flight"
              tone={data.open_deployments.length ? "live" : "quiet"}
            />
          </div>
        </Panel>
      </aside>
    </div>
  );
}

export function PathPage() {
  const q = useOverview();
  return (
    <>
      <PageHeader
        title="Path to production"
        lead="Every buyer model between the training pipeline and full traffic: which gate stopped it, what is in canary, and what production can roll back to."
      />
      <QueryView
        query={q}
        skeleton={<SkeletonBlock label="Loading the path" rows={5} />}
        isEmpty={(d) => d.recent.length === 0 && d.runs.length === 0}
        empty={
          <EmptyState title="Nothing on the path yet">
            Run a continuous-training pipeline (<code>ptp pipeline run</code>) or a simulated season (
            <code>ptpsim run scenarios/dev/smoke.yaml</code>).
          </EmptyState>
        }
      >
        {(d) => <Body data={d} />}
      </QueryView>
    </>
  );
}
