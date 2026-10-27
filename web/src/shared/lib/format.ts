const dateTime = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  timeZone: "UTC",
});

/** Registry times are seconds since the epoch, shown in UTC (the pipelines' clock). */
export const when = (ts: number | null | undefined) =>
  ts ? `${dateTime.format(new Date(ts * 1000))} UTC` : "–";

export const pct = (x: number | null | undefined, digits = 1) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`;

export const share = (x: number) => `${Math.round(x * 100)}%`;

export function duration(from: number, to: number) {
  const h = (to - from) / 3600;
  if (h * 60 < 1) return "under a minute";
  return h < 1
    ? `${Math.round(h * 60)} min`
    : h < 48
      ? `${h.toFixed(h < 10 ? 1 : 0)} h`
      : `${Math.round(h / 24)} days`;
}

/** "ranker_r0007" -> "r0007": the version a routing entry points at. */
export const versionOf = (profile: string | null | undefined) =>
  profile ? profile.replace(/^ranker_/, "") : "";

/** Gate labels as one phrase: "Not worse than production, not worse in any market". */
export const listOfLabels = (labels: string[]) =>
  labels.map((l, i) => (i === 0 ? l : l.charAt(0).toLowerCase() + l.slice(1))).join(", ");
