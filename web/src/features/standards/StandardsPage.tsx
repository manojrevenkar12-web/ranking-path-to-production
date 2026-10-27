import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../../shared/api/client";
import type { StandardsReport } from "../../shared/api/types";
import { REFRESH_MS, STATUS } from "../../shared/theme/tokens";
import { Badge, EmptyState, PageHeader, Panel, QueryView, SkeletonBlock } from "../../shared/ui";
import styles from "./Standards.module.css";

const useStandards = () =>
  useQuery({
    queryKey: ["standards"],
    queryFn: () => api.get<StandardsReport>("/api/standards"),
    refetchInterval: REFRESH_MS,
  });

export function StandardsPage() {
  const q = useStandards();
  return (
    <>
      <PageHeader
        title="Lifecycle standards"
        lead="What every model owes before and while it serves, from config/standards.yaml, checked against the registry. Change the rule there, not the check."
      />
      <QueryView
        query={q}
        skeleton={<SkeletonBlock label="Loading standards" rows={5} />}
        isEmpty={(d) => d.models.length === 0}
        empty={
          <EmptyState title="No models to check">
            Standards are checked once the registry has a version.
          </EmptyState>
        }
      >
        {(d) => {
          const live = d.models.filter((m) => m.status === "production" || m.status === "canary");
          const rest = d.models
            .filter((m) => !(m.status === "production" || m.status === "canary") && !m.conforms)
            .slice(-6);
          const rows = [...live, ...rest];
          return (
            <div className={styles.layout}>
              <Panel title="Models in service, and recent ones short of a standard">
                <div className={styles.scroll}>
                  <table className={styles.matrix}>
                    <thead>
                      <tr>
                        <th scope="col">Version</th>
                        {d.standards.map((s) => (
                          <th scope="col" key={s.id} title={s.rule}>
                            {s.id.replace("_", " ")}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((m) => {
                        const st = STATUS[m.status] ?? { label: m.status, tone: "quiet" as const };
                        return (
                          <tr key={m.version}>
                            <th scope="row">
                              <Link to={`/models/${m.version}`}>{m.version}</Link>{" "}
                              <Badge tone={st.tone}>{st.label}</Badge>
                            </th>
                            {d.standards.map((s) => {
                              const f = m.findings.find((x) => x.standard === s.id);
                              return (
                                <td
                                  key={s.id}
                                  className={f ? styles.short : styles.ok}
                                  title={f?.detail ?? "Meets the standard"}
                                >
                                  {f ? (
                                    <span>{f.detail}</span>
                                  ) : (
                                    <span aria-label="meets the standard">✓</span>
                                  )}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Panel>
              <Panel title="The standards">
                <dl className={styles.list}>
                  {d.standards.map((s) => (
                    <div key={s.id}>
                      <dt>{s.id.replace("_", " ")}</dt>
                      <dd>
                        {s.rule}
                        {s.why && <span className={styles.why}>{s.why}</span>}
                      </dd>
                    </div>
                  ))}
                </dl>
              </Panel>
            </div>
          );
        }}
      </QueryView>
    </>
  );
}
