import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { type FormEvent, useState } from "react";

import { ApiError, api, session } from "../../shared/api/client";
import type { Route, RoutingReport } from "../../shared/api/types";
import { share, versionOf, when } from "../../shared/lib/format";
import { MOTION, REFRESH_MS, SURFACE } from "../../shared/theme/tokens";
import { Button, EmptyState, PageHeader, Panel, QueryView, SkeletonBlock } from "../../shared/ui";
import styles from "./Routing.module.css";

const useRouting = () =>
  useQuery({
    queryKey: ["routing"],
    queryFn: () => api.get<RoutingReport>("/api/routing"),
    refetchInterval: REFRESH_MS,
  });

function RollbackForm({ route, onDone }: { route: Route; onDone: () => void }) {
  const qc = useQueryClient();
  const [token, setToken] = useState(session.token() ?? "");
  const [reason, setReason] = useState("");
  const m = useMutation({
    mutationFn: () => {
      session.setToken(token);
      return api.post<{ production: string }>(`/api/routing/${route.surface}/rollback`, { reason });
    },
    onSuccess: () => {
      void qc.invalidateQueries();
      onDone();
    },
  });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    m.mutate();
  };
  const err = m.error instanceof ApiError ? m.error : null;
  return (
    <form className={styles.form} onSubmit={submit}>
      <p>
        Production goes back to <strong>{versionOf(route.previous)}</strong>, which is still loaded: one
        routing change, no redeploy. {versionOf(route.production)} is marked rolled back.
      </p>
      <label htmlFor={`reason-${route.surface}`}>Why</label>
      <textarea
        id={`reason-${route.surface}`}
        required
        rows={2}
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder="What the on-call saw"
      />
      <label htmlFor={`token-${route.surface}`}>API token</label>
      <input
        id={`token-${route.surface}`}
        type="password"
        autoComplete="off"
        required
        value={token}
        onChange={(e) => setToken(e.target.value)}
      />
      {m.isError && (
        <p className={styles.error} role="alert">
          {err?.status === 401
            ? "That token was not accepted."
            : (err?.detail ?? "The rollback did not go through.")}
        </p>
      )}
      <div className={styles.actions}>
        <Button type="submit" variant="danger" busy={m.isPending} disabled={!reason.trim() || !token.trim()}>
          Roll back {SURFACE[route.surface]?.label.toLowerCase()}
        </Button>
        <Button onClick={onDone}>Cancel</Button>
      </div>
    </form>
  );
}

function SurfaceRouting({ route }: { route: Route }) {
  const [open, setOpen] = useState(false);
  const label = SURFACE[route.surface]?.label ?? route.surface;
  return (
    <Panel title={label} aside={<span className={styles.updated}>changed {when(route.updated_at)}</span>}>
      <div className={styles.split} aria-label={`${label} traffic`}>
        <span className={styles.prod} style={{ flexGrow: 1 - route.share }}>
          {versionOf(route.production)} <small>{share(1 - route.share)}</small>
        </span>
        {route.canary && (
          <span className={styles.canary} style={{ flexGrow: route.share }}>
            {versionOf(route.canary)} <small>{share(route.share)} interleaved</small>
          </span>
        )}
      </div>
      <p className={styles.previous}>
        {route.previous ? (
          <>
            Rollback target: <strong>{versionOf(route.previous)}</strong>
          </>
        ) : (
          "No previous model loaded: a rollback is not possible yet."
        )}
      </p>
      {route.previous && !open && <Button onClick={() => setOpen(true)}>Roll back production</Button>}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: MOTION.quick }}
          >
            <RollbackForm route={route} onDone={() => setOpen(false)} />
          </motion.div>
        )}
      </AnimatePresence>
    </Panel>
  );
}

export function RoutingPage() {
  const q = useRouting();
  return (
    <>
      <PageHeader
        title="Routing"
        lead="Which model serves each surface. Production, canary and the rollback target are all loaded at once, so every change here is a routing change, not a deploy."
      />
      <QueryView
        query={q}
        skeleton={<SkeletonBlock label="Loading routing" rows={3} height="var(--space-8)" />}
        isEmpty={(d) => d.current.length === 0}
        empty={
          <EmptyState title="No routing yet">
            Both surfaces serve the popularity profile until a model is adopted or promoted.
          </EmptyState>
        }
      >
        {(d) => (
          <div className={styles.layout}>
            <div className={styles.surfaces}>
              {d.current.map((r) => (
                <SurfaceRouting key={r.surface} route={r} />
              ))}
            </div>
            <Panel title="Changes">
              <ol className={styles.log}>
                {d.log.map((l) => (
                  <li key={l.id}>
                    <span className={styles.when}>{when(l.at)}</span>
                    <span>
                      <strong>{SURFACE[l.surface]?.label ?? l.surface}</strong>: {l.reason}
                    </span>
                  </li>
                ))}
              </ol>
            </Panel>
          </div>
        )}
      </QueryView>
    </>
  );
}
