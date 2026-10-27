import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";

import type { Gate } from "../../shared/api/types";
import { MOTION, gateLabel } from "../../shared/theme/tokens";
import { Badge } from "../../shared/ui";
import styles from "./Model.module.css";

const STAGE_LABEL: Record<string, string> = {
  data: "Data gates, on the training window",
  model: "Offline gates, against production",
  export: "Export gates",
  serving: "Serving gates, on staging",
  canary: "Canary",
};

function verdict(g: Gate): { label: string; tone: "pass" | "stop" | "warn" } {
  if (g.passed) return { label: "Passed", tone: "pass" };
  if (g.waived_by) return { label: "Waived", tone: "warn" };
  return g.severity === "block" ? { label: "Blocked", tone: "stop" } : { label: "Warning", tone: "warn" };
}

function GateRow({ g }: { g: Gate }) {
  const [open, setOpen] = useState(false);
  const v = verdict(g);
  const id = `gate-${g.id}`;
  return (
    <li className={styles.gate}>
      <button
        type="button"
        className={styles.gateHead}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
      >
        <Badge tone={v.tone}>{v.label}</Badge>
        <span className={styles.gateName}>{gateLabel(g.gate)}</span>
        <span className={styles.gateSummary}>{g.summary}</span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            className={styles.evidence}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: MOTION.move, ease: MOTION.ease }}
          >
            {g.waived_by && <p className={styles.waiver}>Waived by {g.waived_by}</p>}
            <p className={styles.gateId}>
              Gate <code>{g.gate}</code>, {g.severity === "block" ? "blocks the release" : "warns"}
            </p>
            <pre>{JSON.stringify(g.evidence, null, 2)}</pre>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

export function GateList({ gates }: { gates: Gate[] }) {
  const stages = ["data", "model", "export", "serving", "canary"].filter((s) =>
    gates.some((g) => g.stage === s),
  );
  return (
    <div className={styles.stages}>
      {stages.map((s) => {
        const mine = gates.filter((g) => g.stage === s);
        const blocked = mine.filter((g) => g.blocking).length;
        return (
          <section key={s} className={styles.stage} aria-labelledby={`stage-${s}`}>
            <h3 id={`stage-${s}`}>
              {STAGE_LABEL[s] ?? s}
              <span className={styles.stageCount}>
                {blocked ? `${blocked} blocking` : `${mine.length} checked`}
              </span>
            </h3>
            <ul className={styles.gates}>
              {mine.map((g) => (
                <GateRow key={g.id} g={g} />
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
