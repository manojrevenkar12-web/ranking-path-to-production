import type { Interval } from "../../shared/api/types";
import { pct } from "../../shared/lib/format";
import styles from "./Model.module.css";

/**
 * The candidate against production, overall and per market: each row is the paired-bootstrap 95% interval of the
 * relative difference, with zero and the non-inferiority margin marked. A row whose interval reaches left of its
 * margin is what stopped the model.
 */
export function IntervalPlot({
  overall,
  markets,
  margin,
  marketMargin,
  minQueries,
}: {
  overall: Interval;
  markets: Record<string, Interval>;
  margin: number;
  marketMargin: number;
  minQueries: number;
}) {
  const rows = [
    { name: "All markets", r: overall, m: margin },
    ...Object.entries(markets).map(([name, r]) => ({ name, r, m: marketMargin })),
  ];
  const lo = Math.min(-marketMargin * 1.6, ...rows.map((x) => x.r.low));
  const hi = Math.max(marketMargin * 0.8, ...rows.map((x) => x.r.high));
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100;
  return (
    <figure className={styles.plot}>
      <figcaption className="visually-hidden">
        Relative difference from production with 95% intervals, overall and per market
      </figcaption>
      <div className={styles.plotRows}>
        {rows.map(({ name, r, m }) => {
          const thin = name !== "All markets" && r.n < minQueries;
          const fails = !thin && r.low <= -m;
          return (
            <div key={name} className={styles.plotRow}>
              <span className={styles.plotName}>{name}</span>
              <div
                className={styles.plotTrack}
                role="img"
                aria-label={`${name}: ${pct(r.delta)}, interval ${pct(r.low)} to ${pct(r.high)}${thin ? ", too few queries to judge" : ""}`}
              >
                <span className={styles.zero} style={{ left: `${x(0)}%` }} />
                <span
                  className={styles.margin}
                  style={{ left: `${x(-m)}%` }}
                  title={`margin ${pct(-m, 0)}`}
                />
                <span
                  className={`${styles.span} ${fails ? styles.spanStop : thin ? styles.spanThin : styles.spanPass}`}
                  style={{ left: `${x(r.low)}%`, width: `${Math.max(0.6, x(r.high) - x(r.low))}%` }}
                />
                <span className={styles.point} style={{ left: `${x(r.delta)}%` }} />
              </div>
              <span className={styles.plotValue}>
                {pct(r.delta)} <span className={styles.plotN}>{r.n} queries</span>
              </span>
            </div>
          );
        })}
      </div>
    </figure>
  );
}
