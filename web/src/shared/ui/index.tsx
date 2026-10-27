import type { UseQueryResult } from "@tanstack/react-query";
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { ApiError } from "../api/client";
import type { Tone } from "../theme/tokens";
import styles from "./ui.module.css";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

export function Button({
  variant = "default",
  busy,
  children,
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "primary" | "danger"; busy?: boolean }) {
  return (
    <button
      type="button"
      {...rest}
      className={cx(styles.button, variant !== "default" && styles[variant], className)}
      disabled={busy || rest.disabled}
      aria-busy={busy || undefined}
    >
      {children}
    </button>
  );
}

export function Badge({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={cx(styles.badge, styles[`tone_${tone}`])} title={title}>
      {children}
    </span>
  );
}

export function Skeleton({ height = "var(--space-4)", width = "100%" }: { height?: string; width?: string }) {
  return <div className={styles.skeleton} style={{ height, width }} aria-hidden />;
}

/** A labelled loading placeholder shaped like the content it stands in for. */
export function SkeletonBlock({
  label,
  rows = 4,
  height = "var(--space-7)",
}: {
  label: string;
  rows?: number;
  height?: string;
}) {
  return (
    <div className={styles.skeletons} role="status" aria-label={label}>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} height={height} width={i === rows - 1 ? "64%" : "100%"} />
      ))}
    </div>
  );
}

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <section className={styles.state} aria-live="polite">
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </section>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  const e = error instanceof ApiError ? error : null;
  return (
    <section className={cx(styles.state, styles.error)} role="alert">
      <h2>{e?.title ?? "This view could not load"}</h2>
      <p>{e?.detail ?? (error instanceof Error ? error.message : "Something went wrong while loading.")}</p>
      <Button onClick={onRetry}>Try again</Button>
    </section>
  );
}

/**
 * Every view's states before it shows data: loading (a skeleton shaped like the view), failed (what happened and
 * a retry), empty (what it means). Data already shown stays while it refreshes.
 */
export function QueryView<T>({
  query,
  skeleton,
  isEmpty,
  empty,
  children,
}: {
  query: UseQueryResult<T>;
  skeleton: ReactNode;
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) return <>{skeleton}</>;
  if (query.isError && query.data === undefined)
    return <ErrorState error={query.error} onRetry={() => void query.refetch()} />;
  const data = query.data as T;
  if (isEmpty?.(data)) return <>{empty}</>;
  return <>{children(data)}</>;
}

export function PageHeader({
  title,
  lead,
  children,
}: {
  title: string;
  lead?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className={styles.pageHeader}>
      <div>
        <h1>{title}</h1>
        {lead && <p className={styles.lead}>{lead}</p>}
      </div>
      {children}
    </header>
  );
}

export function Panel({
  title,
  aside,
  children,
  className,
}: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx(styles.panel, className)}>
      {(title || aside) && (
        <div className={styles.panelHead}>
          {title && <h2>{title}</h2>}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ value, label, tone }: { value: ReactNode; label: string; tone?: Tone }) {
  return (
    <div className={styles.stat}>
      <span className={cx(styles.statValue, tone && styles[`text_${tone}`])}>{value}</span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  );
}
