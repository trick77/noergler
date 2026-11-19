import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { Fixes, Metrics, Totals } from "./api";
import type { ReactNode } from "react";
import {
  FluidChart,
  Sparkline,
  labelEvery,
  stackTops,
  type Series,
  type Trend,
} from "./Chart";
import {
  calendarDay,
  change,
  compact,
  dayMonth,
  isWeekend,
  money,
  moneyTick,
  num,
  sortedByName,
  weekdayDay,
  windowTitle,
} from "./format";
import { usePoll } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Sep,
  Tile,
  Tiles,
  column,
  eyebrow,
  h2,
  page,
  pageTitle,
  FilterTrack,
  table,
  td,
  tdNum,
  hint,
  th,
  thNum,
} from "./ui";

/** Which chart the pointer is on and which day: every chart draws its
 *  crosshair on that day, only the one under the pointer its tip. */
type Hover = { chart: string; day: number } | null;

// Fixed order, never cycled: a team keeps its colour as teams come and go,
// so a filter that drops one does not repaint the others.
const SERIES_COLORS = [
  "var(--color-s1)",
  "var(--color-s2)",
  "var(--color-s3)",
  "var(--color-s4)",
];

/** days lists every date in [since, until), so a day with no runs is a zero
 *  rather than a gap the chart closes over. The API returns only days that
 *  have rows, deliberately: filling them is the caller's job, and only the
 *  window itself knows how wide it is: 30 rolling days here, a calendar
 *  month (28 to 31) for a request without ?days=.
 *
 *  The window's ends are the SERVER's days, so they are read off the ISO
 *  string rather than through the browser's clock. `new Date(iso)` plus
 *  local getters re-derives the day in the VIEWER's zone: a +02:00 server's
 *  September renders as Aug 31 to Sep 29 for a reader in New York, titles
 *  itself "August", and drops Sep 30 off the chart. The bucket keys the
 *  server sends (date_trunc, formatted server-side) would then miss by one
 *  the whole way down.
 *
 *  UTC arithmetic in the middle, because Date.UTC has no DST: stepping a
 *  local date across a spring-forward boundary can land on the same day
 *  twice. */
function days(since: string, until: string): string[] {
  const start = calendarDay(since);
  const end = calendarDay(until);
  if (start === null || end === null) return [];

  const out: string[] = [];
  const d = new Date(start);
  // The server caps a window at 90 days; the bound stops a bad pair spinning.
  for (let i = 0; d.getTime() < end && i < 400; i++) {
    out.push(d.toISOString().slice(0, 10));
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return out;
}

/** The windows on offer: 30 days by default, 7 and 90 (the server's cap) on
 *  request (user, 2026-11-10). In the URL (?days=7), so a view can be
 *  shared; 30 is the bare path, and anything else reads as 30. */
const SPANS = [7, 30, 90] as const;
type Span = (typeof SPANS)[number];
function readSpan(): Span {
  const raw = Number(new URLSearchParams(window.location.search).get("days"));
  return SPANS.find((s) => s === raw) ?? 30;
}
/** writeSpan puts the span in the URL, touching only ?days=: 30 is its
 *  absence, and any other parameter stays as it was. */
function writeSpan(span: Span) {
  const params = new URLSearchParams(window.location.search);
  if (span === 30) params.delete("days");
  else params.set("days", String(span));
  const search = params.toString();
  const url = window.location.pathname + (search ? `?${search}` : "");
  if (window.location.pathname + window.location.search !== url) {
    window.history.replaceState({}, "", url);
  }
}

/** The Activity card's kinds, one view each (user, 2026-11-17): a new
 *  figure is one more kind here, never another chart or column. Findings
 *  the default; the choice rides the URL (?activity=tasks) like the span. */
const KINDS = ["findings", "tasks", "fixes", "engagements"] as const;
type Kind = (typeof KINDS)[number];
function readKind(): Kind {
  const raw = new URLSearchParams(window.location.search).get("activity");
  return KINDS.find((k) => k === raw) ?? "findings";
}
/** writeKind puts the kind in the URL, touching only ?activity=. */
function writeKind(kind: Kind) {
  const params = new URLSearchParams(window.location.search);
  if (kind === "findings") params.delete("activity");
  else params.set("activity", kind);
  const search = params.toString();
  const url = window.location.pathname + (search ? `?${search}` : "");
  if (window.location.pathname + window.location.search !== url) {
    window.history.replaceState({}, "", url);
  }
}
/** The server's kinds behind each view. */
const SERVER_KINDS: Record<Kind, string[]> = {
  findings: ["finding"],
  tasks: ["task"],
  fixes: ["fix_committed", "fix_other"],
  engagements: ["engagement"],
};

/** trailingMean is each day's mean over itself and the six days before.
 *  Seven, because every such week holds exactly one weekend: a shorter
 *  window dipped every Saturday. The caller passes the six lead days the
 *  API sends before the window (lead_since) and drops them afterwards, so
 *  the line starts on the window's first day. A day with fewer than six
 *  days behind it (an API without the lead) is null rather than a partial
 *  week passed off as a whole one. */
function trailingMean(values: number[], n = 7): (number | null)[] {
  return values.map((_, i) =>
    i < n - 1
      ? null
      : values.slice(i - n + 1, i + 1).reduce((a, b) => a + b, 0) / n,
  );
}

/** niceTicks picks 0 / half / max, rounded up so the top tick is a round
 *  number rather than the data's own maximum. */
function niceTicks(max: number): number[] {
  if (max <= 0) return [0, 1];
  const mag = Math.pow(10, Math.floor(Math.log10(max)));
  const top = Math.ceil(max / (mag / 2)) * (mag / 2);
  return [0, top / 2, top];
}

/** wholeTicks lifts a count axis whose middle tick is fractional to the
 *  next even top: [0, 7.5, 15] becomes [0, 8, 16]. */
function wholeTicks(ticks: number[]): number[] {
  const mid = ticks[1];
  if (Number.isInteger(mid)) return ticks;
  const half = Math.ceil(mid);
  return [0, half, half * 2];
}

export function MetricsPage() {
  const { t } = useTranslation();
  // Rolling days, today included (user's call): the trend reads the same on
  // the 1st as on the 30th. The cost no longer matches the monthly bill.
  const [span, setSpan] = useState<Span>(readSpan);
  const { data, failed, stale } = usePoll<Metrics>(
    `metrics?days=${span}`,
    60000,
  );
  const pick = setSpan;
  // The URL follows the span. A link naming a window not on offer (?days=14) shows 30; the address
  // bar says so too, or a copied link would name a window the page never
  // showed. An effect, not the state initializer: render stays pure.
  useEffect(() => writeSpan(span), [span]);
  const [kind, setKind] = useState<Kind>(readKind);
  useEffect(() => writeKind(kind), [kind]);
  const [hover, setHover] = useState<Hover>(null);
  // A series id (a team's slug, or the folded tail) the legends and the
  // team table bring forward in every chart that draws it.
  const [focus, setFocus] = useState<string | null>(null);

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={pageTitle}>{t("nav.metrics")}</h2>
          {failed ? (
            <Failed what={t("metrics.failedWhat")} />
          ) : (
            <div className="skeleton h-24 rounded-ui" />
          )}
        </div>
      </div>
    );
  }

  const window = days(data.since, data.until);
  const title = windowTitle(data);
  // A new window is a new chart: keyed by it, the charts draw in afresh
  // when the span's data arrives, rather than reusing day 1-7's bars for
  // other days and growing only the rest. A poll of the same window keeps
  // them.
  const drawKey = `${data.since}|${data.until}`;
  // Day-of-month labels, as many as the chart's width holds without two
  // colliding. Dates rather than an offset from today: "d-13" says nothing
  // on its own. A rolling window crosses a month, where plain days read 26,
  // 30, 3, 7: the first label and the first of each new month carry the
  // month name.
  //
  // A rolling window's last day is today, still filling: drawn as such
  // (outlined bars, a dashed last segment) and labelled "today", with no
  // date label crowding it.
  const partial = data.window === "rolling";
  const labels = (width: number) => {
    const step = labelEvery(width, window.length);
    return window.map((day, i) => {
      const last = window.length - 1;
      if (partial && i === last) return t("metrics.today");
      if (i % step !== 0 || (partial && last - i < step)) return "";
      const prev = i - step;
      if (prev < 0 || window[prev].slice(0, 7) !== day.slice(0, 7))
        return dayMonth(day);
      return String(Number(day.slice(-2)));
    });
  };
  const weekends = window.map(isWeekend);
  /** What every chart on the page shares: the weekend bands, today, the
   *  crosshair's day and the focused team. */
  const shared = (chart: string) => ({
    bands: weekends,
    partialLast: partial,
    // Every chart keeps the trend label's room, trend or not, so two
    // stacked in a column put each day at the same x.
    reserveTrend: true,
    hover: hover?.day ?? null,
    // The same day again keeps the state as it is: a new object on every
    // pointer move re-rendered the whole page per pixel.
    onHover: (day: number | null) =>
      setHover((h) =>
        day === null
          ? null
          : h?.chart === chart && h.day === day
            ? h
            : { chart, day },
      ),
    focus,
  });

  // Cost per day per team. A team's missing day is 0 for the chart's
  // purposes, which is true: it spent nothing.
  //
  // Team order comes from by_team, which the server sorts by spend. Taking
  // it from data.daily instead ordered by whichever team happened to appear
  // first on the earliest day, which is unspecified within a day: the same
  // data could repaint every series a different colour between two polls.
  const ranked = data.by_team.map((row) => row.team);
  const charted = ranked.slice(0, SERIES_COLORS.length - 1);
  const rest = ranked.slice(SERIES_COLORS.length - 1);

  // Series are matched by SLUG and labelled by name: data.daily is keyed by
  // slug, and two teams may well display the same name.
  const nameOf = (slug: string) =>
    data.by_team.find((row) => row.team === slug)?.team_name ?? slug;

  /** perDay sums `count` over the rows of each window day that `keep` lets
   *  through. */
  const perDayIn = <R extends { day: string }>(
    on: string[],
    rows: R[],
    count: (r: R) => number,
    keep: (r: R) => boolean = () => true,
  ) =>
    on.map((day) =>
      rows
        .filter((r) => r.day === day && keep(r))
        .reduce((n, r) => n + count(r), 0),
    );
  const perDay = <R extends { day: string }>(
    rows: R[],
    count: (r: R) => number,
    keep?: (r: R) => boolean,
  ) => perDayIn(window, rows, count, keep);
  const spend = (r: Metrics["daily"][number]) =>
    r.cost_usd ? Number(r.cost_usd) : 0;

  const lineOf = (r: Metrics["daily"][number]) => r.lines;

  /** teamSeriesOf is one series per charted team plus the folded tail,
   *  over the rows keep lets through. Cost, LOC and Activity share the
   *  ranking and the colours, so a team reads the same hue on every chart. */
  const teamSeriesOf = <R extends { day: string; team: string }>(
    rows: R[],
    count: (r: R) => number,
    keep: (r: R) => boolean = () => true,
  ) => {
    const out: Series[] = charted.map((team, i) => ({
      id: team,
      name: nameOf(team),
      color: SERIES_COLORS[i],
      values: perDay(rows, count, (r) => keep(r) && r.team === team),
    }));
    // The tail folds into one band rather than being dropped. Truncating it
    // let the card say "No priced runs in this window" beside a Cost tile
    // showing the server's full, non-zero total. Every team past the
    // charted ones, a team the ranking does not know included; named by
    // the teams it holds.
    const inTail = (r: R) => keep(r) && !charted.includes(r.team);
    const tailTeams = [
      ...new Set([...rest, ...rows.filter(inTail).map((r) => r.team)]),
    ];
    const tail = perDay(rows, count, inTail);
    if (rest.length > 0 || tail.some((v) => v > 0)) {
      out.push({
        // Not a slug: TeamSlugRE cannot start with "_", so no team collides.
        id: "__rest",
        name:
          tailTeams.length === 1
            ? nameOf(tailTeams[0])
            : t("metrics.more", { count: tailTeams.length }),
        color: SERIES_COLORS[SERIES_COLORS.length - 1],
        values: tail,
      });
    }
    return out;
  };
  const teamSeries = (count: (r: Metrics["daily"][number]) => number) =>
    teamSeriesOf(data.daily, count);
  const costSeries = teamSeries(spend);
  const linesSeries = teamSeries(lineOf);
  const costTop = Math.max(
    ...window.map((_, i) =>
      costSeries.reduce((sum, s) => sum + s.values[i], 0),
    ),
    0,
  );

  // Attempts per day, by outcome. Skipped and failed use the status hues,
  // not the series ramp: they are states, not categories.
  const bucket = (test: (o: string) => boolean) =>
    perDay(
      data.daily_attempts,
      (b) => b.count,
      (b) => test(b.outcome),
    );
  const runSeries: Series[] = [
    {
      // "__", like the folded tail: a team slug cannot start with it, so
      // focusing a team named "failed" never lights up an outcome.
      id: "__reviewed",
      name: t("metrics.seriesReviewed"),
      color: "var(--color-s1)",
      values: bucket((o) => o === "ok"),
    },
    {
      id: "__skipped",
      name: t("metrics.seriesSkipped"),
      color: "var(--color-ochre)",
      values: bucket((o) => o === "skipped"),
    },
    {
      id: "__failed",
      name: t("metrics.seriesFailed"),
      color: "var(--color-danger)",
      values: bucket((o) => o !== "ok" && o !== "skipped"),
    },
  ];
  const runTotals = window.map((_, i) =>
    runSeries.reduce((sum, s) => sum + s.values[i], 0),
  );
  const runTop = Math.max(...runTotals, 0);

  const costTotals = costSeries.length
    ? window.map((_, i) => costSeries.reduce((sum, s) => sum + s.values[i], 0))
    : [];

  // Cost trends its TOTAL, every team summed: a trend of one team would
  // read as the trend of the chart. Throughput trends REVIEWED runs only
  // (user, 2026-11-10, reversing the earlier total): skips and failures are
  // not throughput, and its legend says so.
  //
  // Summed straight off the rows over lead + window, not off the series:
  // a team that spent only in the lead days is in no series, and its spend
  // still belongs in the week the window's first days average over.
  const lead = days(data.lead_since, data.since);
  const trended = [...lead, ...window];
  const ofWindow = (values: number[]) =>
    trailingMean(values).slice(lead.length);
  const trendColor = "var(--color-ink)";
  const runTrend: Trend = {
    values: ofWindow(
      perDayIn(
        trended,
        data.daily_attempts,
        (b) => b.count,
        (b) => b.outcome === "ok",
      ),
    ),
    color: trendColor,
    format: (v) => num(Math.round(v)),
  };
  const costTrend: Trend = {
    values: ofWindow(perDayIn(trended, data.daily, spend)),
    color: trendColor,
    format: (v) => money(v.toFixed(6)),
  };
  // Lines, not a stack: each team at its own value. Stacked, every stroke
  // sat on the cumulative top, so the last team traced the daily total and
  // read as the biggest spender. The axis fits the highest team or the
  // trend, whichever is higher: the trend averages the sum, so it can
  // clear every single team.
  const costTicks = niceTicks(
    Math.max(
      ...costSeries.flatMap((s) => s.values),
      ...costTrend.values.map((v) => v ?? 0),
      0,
    ),
  );
  // The runs trend averages the lead days too, so a busy week before the
  // window can put it above every bar in it.
  const runTicks = niceTicks(
    Math.max(runTop, ...runTrend.values.map((v) => v ?? 0)),
  );
  // LOC like cost: a line per team, the trend averaging every team summed.
  // Days before the figure existed read 0 and pull the trend down. The
  // card's note names such runs in the window; the six lead days have no
  // per-day count, so their share goes unnamed until a week after 0029.
  const linesTrend: Trend = {
    values: ofWindow(perDayIn(trended, data.daily, lineOf)),
    color: trendColor,
    format: (v) => compact(Math.round(v)),
  };
  const linesTop = Math.max(...linesSeries.flatMap((s) => s.values), 0);
  // Whole lines on the axis: niceTicks halves the top, and 15 LOC would
  // put a 7.5 tick under a count.
  const linesTicks = wholeTicks(
    niceTicks(Math.max(linesTop, ...linesTrend.values.map((v) => v ?? 0))),
  );
  // Every run predates the figure: nothing was counted, which is not
  // "no lines reviewed". The tile shows a dash for the same case.
  const linesNone =
    data.totals.runs > 0 &&
    data.totals.lines_uncounted_runs >= data.totals.runs;

  // Activity (user, 2026-11-17): one card, one kind at a time, never two
  // kinds in one bar: a finding and a task are never summed, and findings
  // run ten times higher than the rest. Each kind stacks by team, the same
  // ranking and hues as Cost and LOC, so a team reads the same everywhere.
  //
  // Fixes alone stack by outcome (user, 2026-11-13, kept 2026-11-17): the
  // bar's height is the requests, the green part the commits. States, not
  // teams: status hues, never the team ramp. Not committed is neutral, it
  // mixes refusals, conflicts and "not feasible"; ochre already reads
  // "skipped" in Throughput.
  const acts = data.activity_daily ?? [];
  const actCount = (r: Metrics["activity_daily"][number]) => r.count;
  const ofKind = (kind: string) =>
    perDay(acts, actCount, (r) => r.kind === kind);
  const activitySeries = (k: Kind): Series[] => {
    if (k === "fixes") {
      return [
        {
          id: "__committed",
          name: t("metrics.fixesCommitted"),
          color: "var(--color-online)",
          values: ofKind("fix_committed"),
        },
        {
          id: "__not_committed",
          name: t("metrics.fixesNotCommitted"),
          color: "var(--color-muted)",
          values: ofKind("fix_other"),
        },
      ];
    }
    const kinds = SERVER_KINDS[k];
    return teamSeriesOf(acts, actCount, (r) => kinds.includes(r.kind));
  };
  const actSeries = activitySeries(kind);
  const actTop = Math.max(...stackTops(actSeries).flat(), 0);
  const actTicks = wholeTicks(niceTicks(actTop));
  const kindLabel = t(`metrics.kind.${kind}`);

  // All teams per day, for the tiles' trends.
  const findingsDaily = perDay(data.daily, (b) => b.findings_posted);
  const linesDaily = perDay(data.daily, lineOf);
  const lines = data.totals.lines_added + data.totals.lines_removed;
  const uncounted = data.totals.lines_uncounted_runs;
  /** The exact total and split behind a compact LOC figure, and the older
   *  runs it leaves out, if any. */
  const linesTitle = (r: Totals) => {
    const split = t("metrics.linesSplit", {
      total: num(r.lines_added + r.lines_removed),
      added: num(r.lines_added),
      removed: num(r.lines_removed),
    });
    return r.lines_uncounted_runs > 0
      ? `${split} (${t("metrics.linesUncounted", { count: r.lines_uncounted_runs })})`
      : split;
  };
  /** What came of a team's fix requests, one outcome per line: the
   *  requests count every ask, committed or not (no write permission, not
   *  feasible, ...). */
  const fixesTitle = (f: Fixes) =>
    f.outcomes.length === 0
      ? t("metrics.fixRequestsTitle")
      : f.outcomes
          .map(
            (o) =>
              `${o.count} ${t(`metrics.fixOutcome.${o.outcome}`, { defaultValue: o.outcome })}`,
          )
          .join("\n");
  /** A team row's LOC: a dash when every run predates the figure, ≥ when
   *  some do, so the column never states a low number as exact. */
  const teamLines = (r: Totals) => {
    if (r.lines_uncounted_runs >= r.runs) return "—";
    const n = compact(r.lines_added + r.lines_removed);
    return r.lines_uncounted_runs > 0 ? `≥${n}` : n;
  };
  const hasTrend = (values: number[]) => values.some((v) => v > 0);

  // The change on the window before, as long and cut at the same hour. A
  // figure that is not whole on either side has none: a window with
  // unpriced runs leans low on cost, one with runs from before the LOC
  // count leans low on lines, and either would print a drop that is not.
  // A tile that shows a dash has no change beside it either.
  const prev = data.previous ?? null;
  const changeOf = (now: number, before: number, whole = true) => {
    if (prev === null || !whole) return undefined;
    const c = change(now, before);
    return c === null ? undefined : (
      <span title={t("metrics.changeTitle", { count: window.length })}>
        {c}
      </span>
    );
  };
  const costChange =
    prev &&
    changeOf(
      Number(data.totals.cost_usd ?? 0),
      Number(prev.totals.cost_usd ?? 0),
      data.totals.runs > 0 &&
        data.totals.cost_usd !== null &&
        prev.totals.cost_usd !== null &&
        data.totals.unpriced_runs === 0 &&
        prev.totals.unpriced_runs === 0,
    );
  const reviewedChange =
    prev && changeOf(data.prs.reviewed, prev.prs.reviewed, data.prs.seen > 0);
  const findingsChange =
    prev && changeOf(data.totals.findings_posted, prev.totals.findings_posted);
  const linesChange =
    prev &&
    changeOf(
      lines,
      prev.totals.lines_added + prev.totals.lines_removed,
      data.totals.runs > 0 &&
        uncounted === 0 &&
        prev.totals.lines_uncounted_runs === 0,
    );

  /** tipFor is a chart's hover tip, for the chart under the pointer only:
   *  the day, each series' value, a stack's total, the trend. */
  const tipFor = (
    chart: string,
    series: Series[],
    fmt: (v: number) => string,
    opts: { trend?: Trend; trendLabel?: string; total?: boolean } = {},
  ) =>
    hover?.chart !== chart
      ? undefined
      : (i: number) => {
          const rows: TipRow[] = series.map((s) => ({
            key: s.id,
            swatch: s.color,
            name: s.name,
            value: fmt(s.values[i]),
          }));
          if (opts.total && series.length > 1) {
            rows.push({
              key: "__total",
              name: t("metrics.total"),
              value: fmt(series.reduce((sum, s) => sum + s.values[i], 0)),
            });
          }
          const tv = opts.trend?.values[i];
          if (opts.trend && tv !== null && tv !== undefined) {
            rows.push({
              key: "__trend",
              line: opts.trend.color,
              name: opts.trendLabel ?? t("metrics.trend"),
              value: opts.trend.format(tv),
            });
          }
          const today = partial && i === window.length - 1;
          return (
            <ChartTip
              day={
                weekdayDay(window[i]) + (today ? ` ${t("metrics.soFar")}` : "")
              }
              rows={rows}
            />
          );
        };
  const costValue = (v: number) => money(v.toFixed(6));

  // A team row, a legend entry: the series id it focuses. A team past the
  // charted ones lives in the folded tail.
  const seriesOf = (slug: string) => (charted.includes(slug) ? slug : "__rest");
  const colorOf = (slug: string) => {
    const i = charted.indexOf(slug);
    return SERIES_COLORS[i < 0 ? SERIES_COLORS.length - 1 : i];
  };
  const teamCosts = data.by_team.map((r) =>
    r.cost_usd === null ? 0 : Number(r.cost_usd),
  );
  const maxTeamCost = Math.max(...teamCosts, 0);
  const sumOf = (count: (r: Metrics["by_team"][number]) => number) =>
    data.by_team.reduce((n, r) => n + count(r), 0);

  return (
    <div className={page}>
      <div className={column}>
        {/* The window is named ONCE, here. Repeating it on every tile and
            card says the same thing five times and crowds out what each
            card actually measures. */}
        <div className="mb-6 flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
          <h2 className={h2}>
            {t("nav.metrics")}
            <Sep />
            <span className="text-muted">{title}</span>
          </h2>
          <FilterTrack
            label={t("metrics.span")}
            options={SPANS.map((s) => ({
              value: s,
              label: t("metrics.spanDays", { count: s }),
            }))}
            value={span}
            onChange={pick}
          />
        </div>

        <div className={stale ? "opacity-40 transition-opacity" : ""}>
          <Tiles cols={4}>
            {/* On an instance that has not run yet, "unpriced" would claim the
              gateway declined to price something. Nothing ran; that is a
              dash. "unpriced" is reserved for runs that actually happened
              and came back without a price.
              Unpriced runs are a note on the cost, not a tile (user,
              2026-11-19): the tile read 0 nearly always, and the cost is
              what they make incomplete. */}
            <Tile
              label={t("metrics.cost")}
              value={data.totals.runs === 0 ? "—" : money(data.totals.cost_usd)}
              warn={
                data.totals.unpriced_runs > 0
                  ? t("metrics.unpricedRuns", {
                      count: data.totals.unpriced_runs,
                    })
                  : undefined
              }
              aside={costChange}
            >
              {costTop > 0 && (
                <div className="mt-2">
                  <Sparkline
                    partial={partial}
                    values={costTotals}
                    label={t("metrics.costTrend")}
                  />
                </div>
              )}
            </Tile>
            {/* PRs, not runs (user's call): runs count every push and mention
              again, so a busy PR inflated the figure and one skipped for a
              missing precondition never showed. Reviewed = got an LLM review
              back; seen = touched at all. No runs line under it either
              (user, 2026-11-10): Throughput shows the runs. */}
            <Tile
              label={t("metrics.reviewed")}
              value={
                data.prs.seen === 0 ? (
                  "—"
                ) : (
                  <span
                    title={t("metrics.reviewedOfSeen", {
                      reviewed: data.prs.reviewed,
                      seen: data.prs.seen,
                    })}
                  >
                    {num(data.prs.reviewed)}
                    <span className="text-muted"> / </span>
                    {num(data.prs.seen)}
                  </span>
                )
              }
              note={data.prs.seen === 0 ? undefined : t("metrics.prsNote")}
              aside={reviewedChange}
            >
              {/* A flat line of zeros is a line that says nothing. */}
              {hasTrend(runSeries[0].values) && (
                <div className="mt-2">
                  <Sparkline
                    partial={partial}
                    values={runSeries[0].values}
                    label={t("metrics.reviewedTrend")}
                  />
                </div>
              )}
            </Tile>
            <Tile
              label={t("metrics.findings")}
              value={num(data.totals.findings_posted)}
              aside={findingsChange}
            >
              {hasTrend(findingsDaily) && (
                <div className="mt-2">
                  <Sparkline
                    partial={partial}
                    values={findingsDaily}
                    label={t("metrics.findingsTrend")}
                  />
                </div>
              )}
            </Tile>
            {/* + and − as one figure, labelled LOC and compact (12.3K), not
              KLOC (user, 2026-11-11): a unit of thousands would read 0.005
              on a quiet week. The exact total and split are on hover.
              Only what the prompts carried, each line once per PR: a
              re-review re-reads lines already counted. Runs from before the
              figure existed add nothing: all of them is a dash, some of
              them a note, never a firm number that silently reads low. */}
            <Tile
              label={t("metrics.linesReviewed")}
              aside={linesChange}
              note={
                uncounted > 0 && uncounted < data.totals.runs
                  ? t("metrics.linesUncounted", { count: uncounted })
                  : undefined
              }
              value={
                data.totals.runs === uncounted ? (
                  "—"
                ) : (
                  <span title={linesTitle(data.totals)}>{compact(lines)}</span>
                )
              }
            >
              {hasTrend(linesDaily) && (
                <div className="mt-2">
                  <Sparkline
                    partial={partial}
                    values={linesDaily}
                    label={t("metrics.linesTrend")}
                  />
                </div>
              )}
            </Tile>
          </Tiles>

          {/* Throughput, cost, LOC (user, 2026-11-11), then activity
            (user, 2026-11-13), two to a row from xl up (user, 2026-11-19):
            the four stacked ran four screens long. What each counts per
            day is the card's note, not an eyebrow above it. Every chart is
            one height, so the two in a row end level. */}
          <div className="mt-5 grid grid-cols-[minmax(0,1fr)] gap-x-5 xl:grid-cols-2">
            <Card
              className="min-w-0"
              title={t("metrics.throughput")}
              note={t("metrics.runsPerDay")}
            >
              {runTop === 0 ? (
                <Empty>{t("metrics.noRuns")}</Empty>
              ) : (
                <>
                  <FluidChart
                    key={drawKey}
                    {...shared("runs")}
                    series={runSeries}
                    trend={runTrend}
                    height={CHART_H}
                    mark="bar"
                    ticks={runTicks}
                    xLabels={labels}
                    label={t("metrics.runsChart", { title })}
                    tip={tipFor("runs", runSeries, num, {
                      trend: runTrend,
                      trendLabel: t("metrics.trendReviewed"),
                      total: true,
                    })}
                  />
                  <Legend
                    series={runSeries}
                    trend={trendColor}
                    trendLabel={t("metrics.trendReviewed")}
                  />
                </>
              )}
            </Card>

            <Card
              className="min-w-0"
              title={t("metrics.spend")}
              note={t("metrics.costPerDay")}
            >
              {costTop === 0 ? (
                <Empty>{t("metrics.noPriced")}</Empty>
              ) : (
                <>
                  <FluidChart
                    key={drawKey}
                    {...shared("cost")}
                    series={costSeries}
                    trend={costTrend}
                    height={CHART_H}
                    mark="line"
                    ticks={costTicks}
                    xLabels={labels}
                    format={moneyTick(costTicks)}
                    label={t("metrics.costChart", { title })}
                    tip={tipFor("cost", costSeries, costValue, {
                      trend: costTrend,
                    })}
                  />
                  <Legend
                    series={costSeries}
                    trend={trendColor}
                    focus={focus}
                    onFocus={setFocus}
                  />
                </>
              )}
            </Card>

            <Card
              className="min-w-0"
              title={t("metrics.codeVolume")}
              note={
                uncounted > 0 && !linesNone ? (
                  <>
                    {t("metrics.linesPerDay")}
                    <Sep />
                    {t("metrics.linesUncounted", { count: uncounted })}
                  </>
                ) : (
                  t("metrics.linesPerDay")
                )
              }
            >
              {linesNone ? (
                <Empty>{t("metrics.linesNotCounted")}</Empty>
              ) : linesTop === 0 ? (
                <Empty>{t("metrics.noLines")}</Empty>
              ) : (
                <>
                  <FluidChart
                    key={drawKey}
                    {...shared("lines")}
                    series={linesSeries}
                    trend={linesTrend}
                    height={CHART_H}
                    mark="line"
                    ticks={linesTicks}
                    xLabels={labels}
                    format={compact}
                    label={t("metrics.linesChart", { title })}
                    tip={tipFor("lines", linesSeries, compact, {
                      trend: linesTrend,
                    })}
                  />
                  <Legend
                    series={linesSeries}
                    trend={trendColor}
                    focus={focus}
                    onFocus={setFocus}
                  />
                </>
              )}
            </Card>

            <Card
              className="min-w-0"
              title={t("metrics.activity")}
              note={t(`metrics.kindNote.${kind}`)}
              right={
                <FilterTrack
                  label={t("metrics.activityKind")}
                  options={KINDS.map((k) => ({
                    value: k,
                    label: t(`metrics.kind.${k}`),
                  }))}
                  value={kind}
                  onChange={setKind}
                />
              }
            >
              {actTop === 0 ? (
                <Empty>{t("metrics.noActivity")}</Empty>
              ) : (
                <>
                  {/* Keyed by the kind too: another kind is another chart,
                    drawn in afresh, never one kind's bars grown into the
                    next. */}
                  <FluidChart
                    key={`${drawKey}|${kind}`}
                    {...shared("activity")}
                    series={actSeries}
                    height={CHART_H}
                    mark="bar"
                    ticks={actTicks}
                    xLabels={labels}
                    label={t("metrics.activityChart", {
                      kind: kindLabel,
                      title,
                    })}
                    tip={tipFor("activity", actSeries, num, { total: true })}
                  />
                  <Legend
                    series={actSeries}
                    focus={focus}
                    onFocus={kind === "fixes" ? undefined : setFocus}
                  />
                </>
              )}
            </Card>
          </div>

          <p className={eyebrow}>{t("metrics.byTeam")}</p>
          <Card>
            {data.by_team.length === 0 ? (
              <Empty>{t("metrics.nothingReviewed")}</Empty>
            ) : (
              <div className="overflow-x-auto overscroll-x-contain">
                <table className={table}>
                  <thead>
                    <tr>
                      <th className={th}>{t("metrics.team")}</th>
                      {/* Runs, not "reviewed": the tile counts PRs now and
                        this column still counts runs. */}
                      <th className={thNum}>{t("metrics.runs")}</th>
                      {/* Fixes beside the findings they apply, as two named
                        counts: "3 / 5" read as nothing (user, 2026-11-13).
                        Tasks beside the findings, never in them (user,
                        2026-11-17). Tokens moved into the cost's hover to
                        make room (user, 2026-11-17: "dann hover"). */}
                      <th className={thNum}>{t("metrics.findings")}</th>
                      <th className={thNum}>{t("metrics.tasks")}</th>
                      <th
                        className={thNum}
                        title={t("metrics.fixRequestsTitle")}
                      >
                        {t("metrics.fixRequests")}
                      </th>
                      <th
                        className={thNum}
                        title={t("metrics.fixesCommittedTitle")}
                      >
                        {t("metrics.fixesCommitted")}
                      </th>
                      <th
                        className={thNum}
                        title={t("metrics.kindNote.engagements")}
                      >
                        {t("metrics.kind.engagements")}
                      </th>
                      <th className={thNum}>{t("metrics.lines")}</th>
                      <th className={thNum}>{t("metrics.cost")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* By name, not by spend: see sortedByName. */}
                    {sortedByName(
                      data.by_team,
                      (r) => r.team_name,
                      (r) => r.team,
                    ).map((row) => {
                      const cost =
                        row.cost_usd === null ? 0 : Number(row.cost_usd);
                      return (
                        <tr
                          key={row.team}
                          onMouseEnter={() => setFocus(seriesOf(row.team))}
                          onMouseLeave={() => setFocus(null)}
                          className="hover:bg-active/50"
                        >
                          <td className={td} title={row.team}>
                            {/* The team's hue on every chart, so a row and
                                its line are found together. */}
                            <i
                              aria-hidden
                              className="mr-2 inline-block h-2.5 w-2.5 rounded-[2px] align-[-1px]"
                              style={{ background: colorOf(row.team) }}
                            />
                            {row.team_name}
                          </td>
                          <Count n={row.runs} />
                          <Count n={row.findings_posted} />
                          <Count n={row.tasks ?? 0} />
                          <Count
                            n={row.fixes.requested}
                            title={fixesTitle(row.fixes)}
                          />
                          <Count n={row.fixes.committed} />
                          <Count n={row.engagements ?? 0} />
                          <td className={tdNum + hint} title={linesTitle(row)}>
                            {teamLines(row)}
                          </td>
                          <td
                            className={tdNum + hint}
                            title={`${t("metrics.tokensIn")}: ${num(row.prompt_tokens)}\n${t("metrics.tokensOut")}: ${num(row.completion_tokens)}`}
                          >
                            {/* The share of the window's spend at a glance,
                                against the biggest team. */}
                            {maxTeamCost > 0 && (
                              <span
                                aria-hidden
                                className="mr-2.5 inline-block h-1 w-16 overflow-hidden rounded-full bg-active align-middle"
                              >
                                <span
                                  className="block h-1 rounded-full"
                                  style={{
                                    width: `${(cost / maxTeamCost) * 100}%`,
                                    background: colorOf(row.team),
                                  }}
                                />
                              </span>
                            )}
                            {money(row.cost_usd)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                  {/* Every team summed. Runs, findings, LOC and cost are
                      the server's own totals; the rest add up here, and
                      are whole counts that do. */}
                  {data.by_team.length > 1 && (
                    <tfoot>
                      <tr className="font-medium">
                        <td className={tdFoot}>{t("metrics.allTeams")}</td>
                        <Count n={data.totals.runs} foot />
                        <Count n={data.totals.findings_posted} foot />
                        <Count n={sumOf((r) => r.tasks ?? 0)} foot />
                        <Count n={sumOf((r) => r.fixes.requested)} foot />
                        <Count n={sumOf((r) => r.fixes.committed)} foot />
                        <Count n={sumOf((r) => r.engagements ?? 0)} foot />
                        <td
                          className={tdNumFoot + hint}
                          title={linesTitle(data.totals)}
                        >
                          {teamLines(data.totals)}
                        </td>
                        <td className={tdNumFoot}>
                          {money(data.totals.cost_usd)}
                        </td>
                      </tr>
                    </tfoot>
                  )}
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

/** One chart height for all four, so two cards in a row end level. */
const CHART_H = 200;

const tdFoot = td + " border-t border-border text-ink";
const tdNumFoot = tdNum + " border-t border-border text-ink";

/** Count is a table count; a zero steps back to faint, so the figures
 *  that are there stand out from the ones that are not. */
function Count({
  n,
  title,
  foot = false,
}: {
  n: number;
  title?: string;
  foot?: boolean;
}) {
  const base = foot ? tdNumFoot : tdNum;
  return (
    <td
      className={
        base + (title ? hint : "") + (n === 0 && !foot ? " text-faint" : "")
      }
      title={title}
    >
      {num(n)}
    </td>
  );
}

type TipRow = {
  key: string;
  name: string;
  value: string;
  /** A series' square, or a trend's line; neither for a total. */
  swatch?: string;
  line?: string;
};

/** ChartTip is a chart's hover tip: the day, then one row per figure. */
function ChartTip({ day, rows }: { day: string; rows: TipRow[] }) {
  return (
    <>
      <div className="mb-1 font-medium whitespace-nowrap text-ink">{day}</div>
      {rows.map((r) => (
        <div
          key={r.key}
          className="flex items-center gap-2 leading-[1.6] whitespace-nowrap"
        >
          <i
            aria-hidden
            className={
              "inline-block shrink-0 " +
              (r.line
                ? "h-[2px] w-3.5 rounded-full"
                : "h-2.5 w-2.5 rounded-[2px]")
            }
            style={{ background: r.swatch ?? r.line ?? "transparent" }}
          />
          <span className="mr-3.5 flex-1">{r.name}</span>
          <span className="font-mono text-[11.5px] tabular-nums text-ink">
            {r.value}
          </span>
        </div>
      ))}
    </>
  );
}

/** A legend is always present for two or more series, so identity is never
 *  carried by colour alone; a trend's line is always named, since a lone
 *  series with a second line beside it is two things to tell apart.
 *  Team legends focus a team on hover or keyboard focus: its series steps
 *  forward in every chart, the others back. */
function Legend({
  series,
  trend,
  trendLabel,
  focus = null,
  onFocus,
}: {
  series: Series[];
  trend?: string;
  /** What the trend averages, when it is not the whole chart. */
  trendLabel?: string;
  focus?: string | null;
  onFocus?: (id: string | null) => void;
}) {
  const { t } = useTranslation();
  if (series.length < 2 && !trend) return null;
  const swatch = (s: Series): ReactNode => (
    <i
      aria-hidden
      className="inline-block h-2.5 w-2.5 rounded-[2px]"
      style={{ background: s.color }}
    />
  );
  return (
    <div className="mt-2.5 flex flex-wrap gap-4 text-[12.5px] text-muted">
      {series.map((s) =>
        onFocus ? (
          <button
            key={s.id}
            type="button"
            onMouseEnter={() => onFocus(s.id)}
            onMouseLeave={() => onFocus(null)}
            onFocus={() => onFocus(s.id)}
            onBlur={() => onFocus(null)}
            className={
              "inline-flex items-center gap-1.5 rounded-[4px] transition-opacity duration-[120ms] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent " +
              (focus === s.id
                ? "text-ink"
                : focus !== null && series.some((o) => o.id === focus)
                  ? "opacity-45"
                  : "hover:text-ink")
            }
          >
            {swatch(s)}
            {s.name}
          </button>
        ) : (
          <span key={s.id} className="inline-flex items-center gap-1.5">
            {swatch(s)}
            {s.name}
          </span>
        ),
      )}
      {trend && (
        <span className="inline-flex items-center gap-1.5">
          <i
            aria-hidden
            className="inline-block h-[2px] w-3.5 rounded-full"
            style={{ background: trend }}
          />
          {trendLabel ?? t("metrics.trend")}
        </span>
      )}
    </div>
  );
}

export { days as _days, niceTicks as _niceTicks, wholeTicks as _wholeTicks };
