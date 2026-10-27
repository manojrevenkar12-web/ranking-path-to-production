import { useQuery } from "@tanstack/react-query";

import { api } from "../../shared/api/client";
import type { Season, SeasonRelease } from "../../shared/api/types";
import { pct } from "../../shared/lib/format";
import type { Tone } from "../../shared/theme/tokens";
import { Badge, EmptyState, PageHeader, Panel, QueryView, SkeletonBlock, Stat } from "../../shared/ui";
import styles from "./Evidence.module.css";

const useSeasons = () =>
  useQuery({ queryKey: ["seasons"], queryFn: () => api.get<Season[]>("/api/seasons") });

const TRUTH: Record<string, { label: string; tone: Tone }> = {
  beneficial: { label: "Better", tone: "pass" },
  neutral: { label: "No real change", tone: "quiet" },
  harmful: { label: "Harmful", tone: "stop" },
  broken: { label: "Broken", tone: "stop" },
};

const PLATFORM: Record<string, string> = {
  promoted: "Promoted",
  blocked_pipeline: "Stopped in the pipeline",
  blocked_serving: "Stopped on staging",
  failed_deploy: "Deploy failed",
  rolled_back: "Rolled back in canary",
  in_canary: "In canary at the end",
  not_deployed: "Not deployed",
};

function right(r: SeasonRelease): boolean | null {
  const harmful = r.truth === "harmful" || r.truth === "broken";
  if (harmful) return r.platform !== "promoted";
  if (r.truth === "beneficial")
    return r.platform === "promoted" || r.platform === "in_canary"
      ? true
      : r.platform === "not_deployed"
        ? null
        : false;
  return null;
}

function SeasonPanel({ s }: { s: Season }) {
  const heldout = s.path.includes("heldout");
  return (
    <Panel
      title={s.scenario}
      aside={<Badge tone={heldout ? "live" : "quiet"}>{heldout ? "Held-out, first run" : "Dev"}</Badge>}
    >
      <div className={styles.stats}>
        <Stat
          value={`${s.summary.harmful_promoted}`}
          label={`of ${s.summary.harmful} harmful releases reached production`}
          tone={s.summary.harmful_promoted ? "stop" : "pass"}
        />
        <Stat
          value={`${s.summary.beneficial_promoted}/${s.summary.beneficial}`}
          label="better releases promoted"
          tone="thread"
        />
        <Stat
          value={pct(s.summary.harmful_exposure_share, 2).replace("+", "")}
          label="of sessions served by a harmful release"
        />
        <Stat
          value={
            s.summary.hours_candidate_to_production_median !== null
              ? `${s.summary.hours_candidate_to_production_median} h`
              : "–"
          }
          label="candidate to production, median"
        />
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Release</th>
              <th scope="col">What it really did</th>
              <th scope="col">Worst market</th>
              <th scope="col">What the platform did</th>
            </tr>
          </thead>
          <tbody>
            {s.releases.map((r) => {
              const t = TRUTH[r.truth] ?? { label: r.truth, tone: "quiet" as const };
              const ok = right(r);
              return (
                <tr key={r.version}>
                  <th scope="row">{r.version}</th>
                  <td>
                    <Badge tone={t.tone}>{t.label}</Badge>{" "}
                    {r.truth === "broken" ? "would not load" : pct(r.overall_delta)}
                  </td>
                  <td>{r.worst_market ? `${r.worst_market} ${pct(r.worst_market_delta)}` : "–"}</td>
                  <td className={ok === false ? styles.wrong : ok ? styles.right : undefined}>
                    {PLATFORM[r.platform] ?? r.platform}
                    {r.failed_step ? ` (${r.failed_step.replace("_", " ")})` : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {s.data_blocks.length > 0 && (
        <p className={styles.blocks}>
          The data gates also stopped {s.data_blocks.length} run{s.data_blocks.length > 1 ? "s" : ""} before
          training:{" "}
          {s.data_blocks
            .map((b) => `${b.run} (${b.causes.join(", ") || "no incident: a false stop"})`)
            .join("; ")}
          .
        </p>
      )}
    </Panel>
  );
}

export function EvidencePage() {
  const q = useSeasons();
  return (
    <>
      <PageHeader
        title="Evidence"
        lead="Simulated seasons: what each release would really have done to purchases (from the simulator's truth, which the platform never reads) next to what the platform did with it."
      />
      <QueryView
        query={q}
        skeleton={<SkeletonBlock label="Loading seasons" rows={4} height="var(--space-8)" />}
        isEmpty={(d) => d.length === 0}
        empty={
          <EmptyState title="No season reports">
            Run <code>ptpsim run scenarios/dev/season-a.yaml</code> and copy its evaluation into reports/.
          </EmptyState>
        }
      >
        {(d) => (
          <div className={styles.layout}>
            {d.map((s) => (
              <SeasonPanel key={s.path} s={s} />
            ))}
          </div>
        )}
      </QueryView>
    </>
  );
}
