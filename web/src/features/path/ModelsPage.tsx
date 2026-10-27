import { useState } from "react";
import { Link } from "react-router-dom";

import { pct, when } from "../../shared/lib/format";
import { STATUS } from "../../shared/theme/tokens";
import { Badge, EmptyState, PageHeader, Panel, QueryView, SkeletonBlock } from "../../shared/ui";
import styles from "./Path.module.css";
import { PathRail } from "./PathRail";
import { useModels } from "./api";

const KINDS = [
  { value: "", label: "All" },
  { value: "ranker", label: "Search ranker" },
  { value: "tower", label: "Homepage two-tower" },
];

export function ModelsPage() {
  const [kind, setKind] = useState("");
  const q = useModels(kind || undefined);
  return (
    <>
      <PageHeader
        title="Model versions"
        lead="Every version the registry knows, newest first, with how far it got."
      >
        <div className={styles.filter} role="radiogroup" aria-label="Model">
          {KINDS.map((k) => (
            <label key={k.value} className={kind === k.value ? styles.filterOn : undefined}>
              <input
                type="radio"
                name="kind"
                value={k.value}
                checked={kind === k.value}
                onChange={() => setKind(k.value)}
              />
              {k.label}
            </label>
          ))}
        </div>
      </PageHeader>
      <QueryView
        query={q}
        skeleton={<SkeletonBlock label="Loading versions" rows={6} />}
        isEmpty={(d) => d.length === 0}
        empty={
          <EmptyState title="No versions of this model yet">
            A training pipeline registers one when it finishes, whether or not it passes.
          </EmptyState>
        }
      >
        {(versions) => (
          <Panel>
            <ul className={styles.versionList}>
              {versions.map((v) => {
                const s = STATUS[v.status] ?? { label: v.status, tone: "quiet" as const };
                return (
                  <li key={v.version} className={styles.version}>
                    <div className={styles.versionHead}>
                      <Link to={`/models/${v.version}`} className={styles.versionId}>
                        {v.version}
                      </Link>
                      <Badge tone={s.tone}>{s.label}</Badge>
                      <span className={styles.meta}>{when(v.created_at)}</span>
                      {v.offline && <span className={styles.offline}>{pct(v.offline.delta)}</span>}
                    </div>
                    {v.path && <PathRail path={v.path} version={v.version} size="compact" sew={false} />}
                  </li>
                );
              })}
            </ul>
          </Panel>
        )}
      </QueryView>
    </>
  );
}
