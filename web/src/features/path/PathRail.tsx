import { motion, useReducedMotion } from "framer-motion";

import type { PathStation } from "../../shared/api/types";
import { cx } from "../../shared/ui";
import { listOfLabels } from "../../shared/lib/format";
import { MOTION, STATION_STATE, STATIONS, gateLabel } from "../../shared/theme/tokens";
import styles from "./PathRail.module.css";

/**
 * The path a model version took: six stations joined by a stitched line. The stitch is drawn solid as far as the
 * version got and sewn in once when the view opens; where it stopped there is a notch and the gate that stopped it.
 */
export function PathRail({
  path,
  version,
  size = "full",
  sew = true,
}: {
  path: PathStation[];
  version: string;
  size?: "full" | "compact";
  sew?: boolean;
}) {
  const reduce = useReducedMotion();
  const byId = Object.fromEntries(path.map((p) => [p.station, p]));
  const reached = path.filter(
    (p) => p.state === "passed" || p.state === "active" || p.state === "stopped" || p.state === "waiting",
  ).length;
  const fraction = STATIONS.length > 1 ? Math.max(0, reached - 1) / (STATIONS.length - 1) : 0;
  const stoppedAt = path.find((p) => p.state === "stopped");
  return (
    <div className={cx(styles.rail, styles[size])}>
      <div className={styles.track} aria-hidden>
        <span className={styles.stitch} />
        <motion.span
          className={cx(styles.sewn, stoppedAt && styles.sewnStopped)}
          style={{ width: `${fraction * 100}%` }}
          initial={sew && !reduce ? { scaleX: 0 } : false}
          animate={{ scaleX: 1 }}
          transition={{ duration: MOTION.sew, ease: MOTION.ease }}
        />
      </div>
      <ol className={styles.stations} aria-label={`Path of ${version}`}>
        {STATIONS.map((s) => {
          const st = byId[s.id] ?? { station: s.id, state: "pending", gates: [] };
          const meta = STATION_STATE[st.state];
          const gates = st.gates.map(gateLabel);
          return (
            <li
              key={s.id}
              className={cx(styles.station, styles[st.state])}
              title={`${s.label}: ${meta.label}${gates.length ? ` (${listOfLabels(gates)})` : ""}`}
            >
              <span className={styles.mark} aria-hidden />
              <span className={styles.label}>{s.label}</span>
              <span className="visually-hidden">{meta.label}</span>
              {size === "full" && st.state === "stopped" && gates.length > 0 && (
                <span className={styles.reason}>{listOfLabels(gates)}</span>
              )}
              {size === "full" && st.state === "waiting" && st.note && (
                <span className={styles.reason}>{st.note}</span>
              )}
              {size === "full" && st.waived && <span className={styles.reason}>waived</span>}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
