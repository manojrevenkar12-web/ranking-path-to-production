import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api } from "../../shared/api/client";
import type { Gate, Run } from "../../shared/api/types";
import { duration, when } from "../../shared/lib/format";
import { OUTCOME, REFRESH_MS, gateLabel } from "../../shared/theme/tokens";
import { Badge, EmptyState, PageHeader, Panel, QueryView, SkeletonBlock } from "../../shared/ui";
import { GateList } from "../model/GateList";
import styles from "./Runs.module.css";

const useRuns = () =>
  useQuery({ queryKey: ["runs"], queryFn: () => api.get<Run[]>("/api/runs"), refetchInterval: REFRESH_MS });

type RunDetail = {
  run: Run & { params: Record<string, unknown> };
  gates: Gate[];
  relay: { message_id: string; action: string; received_at: number }[];
};
const useRun = (id: string) =>
  useQuery({
    queryKey: ["run", id],
    queryFn: () => api.get<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  });

export function RunsPage() {
  const q = useRuns();
  return (
    <>
      <PageHeader
        title="Pipeline runs"
        lead="Every continuous-training and batch-scoring run, with the gate that stopped it. A run that ends with a candidate publishes an event that starts the deploy workflow."
      />
      <QueryView
        query={q}
        skeleton={<SkeletonBlock label="Loading runs" rows={8} height="var(--space-6)" />}
        isEmpty={(d) => d.length === 0}
        empty={
          <EmptyState title="No pipeline has run yet">
            Runs appear here when Vertex AI (or the local engine) starts one.
          </EmptyState>
        }
      >
        {(runs) => (
          <Panel>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Started</th>
                  <th scope="col">Outcome</th>
                  <th scope="col">Stopped by</th>
                  <th scope="col">Version</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => {
                  const o = OUTCOME[r.outcome ?? ""] ?? { label: r.state, tone: "quiet" as const };
                  return (
                    <tr key={r.run_id}>
                      <td>
                        <Link to={`/runs/${r.run_id}`}>{r.run_id}</Link>
                      </td>
                      <td>
                        {when(r.started_at)}
                        {r.finished_at && (
                          <span className={styles.took}>{duration(r.started_at, r.finished_at)}</span>
                        )}
                      </td>
                      <td>
                        <Badge tone={o.tone}>{o.label}</Badge>
                      </td>
                      <td className={styles.stopped}>
                        {r.blocking.map(gateLabel).join(", ") || <span className={styles.dash}>–</span>}
                      </td>
                      <td>
                        {r.version ? (
                          <Link to={`/models/${r.version}`}>{r.version}</Link>
                        ) : (
                          <span className={styles.dash}>–</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Panel>
        )}
      </QueryView>
    </>
  );
}

export function RunPage() {
  const { runId = "" } = useParams();
  const q = useRun(runId);
  return (
    <QueryView query={q} skeleton={<SkeletonBlock label={`Loading ${runId}`} rows={6} />}>
      {(d) => {
        const o = OUTCOME[d.run.outcome ?? ""] ?? { label: d.run.state, tone: "quiet" as const };
        return (
          <>
            <PageHeader
              title={d.run.run_id}
              lead={`${d.run.kind} pipeline, ${d.run.engine ?? ""} engine, started ${when(d.run.started_at)}.`}
            >
              <Badge tone={o.tone}>{o.label}</Badge>
            </PageHeader>
            <div className={styles.detail}>
              <Panel title="Gates">
                {d.gates.length ? (
                  <GateList gates={d.gates} />
                ) : (
                  <EmptyState title="No gate ran">The run stopped before its first check.</EmptyState>
                )}
              </Panel>
              <Panel title="Event">
                {d.run.error && <p className={styles.error}>{d.run.error}</p>}
                {d.run.version && (
                  <p>
                    Registered <Link to={`/models/${d.run.version}`}>{d.run.version}</Link>.
                  </p>
                )}
                {d.relay.length === 0 ? (
                  <p className={styles.dash}>No pipeline event reached the relay for this run.</p>
                ) : (
                  <ul className={styles.relay}>
                    {d.relay.map((m) => (
                      <li key={m.message_id}>
                        Message {m.message_id}: <strong>{m.action.replace("_", " ")}</strong>,{" "}
                        {when(m.received_at)}
                      </li>
                    ))}
                  </ul>
                )}
              </Panel>
            </div>
          </>
        );
      }}
    </QueryView>
  );
}
