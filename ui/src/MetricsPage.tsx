import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { Metrics } from "./api";
import { Chart, Sparkline, type Series, type Trend } from "./Chart";
import {
  calendarDay,
  dayMonth,
  money,
  moneyTick,
  tokens,
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
  lede,
  page,
  table,
  td,
  tdNum,
  th,
  thNum,
} from "./ui";

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

/** The windows on offer: 30 days by default, 90 (the server's cap) on
 *  request (user, 2026-11-10). In the URL (?days=90), so a view can be
 *  shared; 30 is the bare path. */
const SPANS = [30, 90] as const;
type Span = (typeof SPANS)[number];
function readSpan(): Span {
  return new URLSearchParams(window.location.search).get("days") === "90"
    ? 90
    : 30;
}
function writeSpan(span: Span) {
  const url = window.location.pathname + (span === 30 ? "" : `?days=${span}`);
  if (window.location.pathname + window.location.search !== url) {
    window.history.replaceState({}, "", url);
  }
}

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

export function MetricsPage() {
  const { t } = useTranslation();
  // Rolling days, today included (user's call): the trend reads the same on
  // the 1st as on the 30th. The cost no longer matches the monthly bill.
  const [span, setSpan] = useState<Span>(readSpan);
  const { data, failed, stale } = usePoll<Metrics>(
    `metrics?days=${span}`,
    60000,
  );
  const pick = (s: Span) => {
    setSpan(s);
    writeSpan(s);
  };

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>{t("nav.metrics")}</h2>
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
  // Day-of-month labels, every few days, so they never collide. Dates rather
  // than an offset from today: "d-13" says nothing on its own. A rolling
  // window crosses a month, where plain days read 26, 30, 3, 7: the first
  // label and the first of each new month carry the month name.
  const step = window.length > 45 ? 7 : window.length > 20 ? 4 : 3;
  const label = (day: string, i: number) => {
    if (i % step !== 0) return "";
    const prev = i - step;
    if (prev < 0 || window[prev].slice(0, 7) !== day.slice(0, 7))
      return dayMonth(day);
    return String(Number(day.slice(-2)));
  };

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

  const costSeries: Series[] = charted.map((team, i) => ({
    id: team,
    name: nameOf(team),
    color: SERIES_COLORS[i],
    values: perDay(data.daily, spend, (r) => r.team === team),
  }));
  // The tail folds into one band rather than being dropped. Truncating it
  // let the card say "No priced runs in this window" beside a Cost tile
  // showing the server's full, non-zero total.
  if (rest.length > 0) {
    costSeries.push({
      // Not a slug: TeamSlugRE cannot start with "_", so no team collides.
      id: "__rest",
      name:
        rest.length === 1
          ? nameOf(rest[0])
          : t("metrics.more", { count: rest.length }),
      color: SERIES_COLORS[SERIES_COLORS.length - 1],
      values: perDay(data.daily, spend, (r) => rest.includes(r.team)),
    });
  }
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
      id: "reviewed",
      name: t("metrics.seriesReviewed"),
      color: "var(--color-s1)",
      values: bucket((o) => o === "ok"),
    },
    {
      id: "skipped",
      name: t("metrics.seriesSkipped"),
      color: "var(--color-ochre)",
      values: bucket((o) => o === "skipped"),
    },
    {
      id: "failed",
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

  // Both charts trend their TOTAL, the whole bar and every team summed:
  // the card's title already says what is counted, and a trend of one
  // segment would read as the trend of the chart (user, 2026-11-10).
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
    values: ofWindow(perDayIn(trended, data.daily_attempts, (b) => b.count)),
    color: trendColor,
    format: (v) => Math.round(v).toLocaleString(),
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

  // All teams per day, for the tiles' trends.
  const findingsDaily = perDay(data.daily, (b) => b.findings_posted);
  const unpricedDaily = perDay(data.daily, (b) => b.unpriced_runs);
  const hasTrend = (values: number[]) => values.some((v) => v > 0);

  return (
    <div className={page}>
      <div className={column}>
        {/* The window is named ONCE, here. Repeating it on every tile and
            card says the same thing five times and crowds out what each
            card actually measures. */}
        <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
          <h2 className={h2}>
            {t("nav.metrics")}
            <Sep />
            <span className="text-muted">{title}</span>
          </h2>
          <span
            className="flex gap-0.5"
            role="group"
            aria-label={t("metrics.span")}
          >
            {SPANS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => pick(s)}
                aria-pressed={span === s}
                className={
                  "rounded-ui-sm px-2 py-0.5 text-[12px] transition-colors " +
                  (span === s
                    ? "bg-panel text-ink"
                    : "text-muted hover:bg-panel hover:text-ink")
                }
              >
                {t("metrics.spanDays", { count: s })}
              </button>
            ))}
          </span>
        </div>
        <p className={lede}>{t("metrics.lede")}</p>

        <div className={stale ? "opacity-40 transition-opacity" : ""}>
          <Tiles>
            {/* On an instance that has not run yet, "unpriced" would claim the
              gateway declined to price something. Nothing ran; that is a
              dash. "unpriced" is reserved for runs that actually happened
              and came back without a price. */}
            <Tile
              label={t("metrics.cost")}
              value={data.totals.runs === 0 ? "—" : money(data.totals.cost_usd)}
            >
              {costTop > 0 && (
                <div className="mt-2">
                  <Sparkline
                    values={costTotals}
                    label={t("metrics.costTrend")}
                  />
                </div>
              )}
            </Tile>
            {/* PRs, not runs (user's call): runs count every push and mention
              again, so a busy PR inflated the figure and one skipped for a
              missing precondition never showed. Reviewed = got an LLM review
              back; seen = touched at all. The runs stay as the line below. */}
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
                    {data.prs.reviewed.toLocaleString()}
                    <span className="text-muted"> / </span>
                    {data.prs.seen.toLocaleString()}
                  </span>
                )
              }
              note={data.prs.seen === 0 ? undefined : t("metrics.prsNote")}
            >
              {data.totals.runs > 0 && (
                <div className="mt-0.5 text-[12.5px] text-muted tabular-nums">
                  {t("metrics.runCount", {
                    count: data.totals.runs,
                    n: data.totals.runs.toLocaleString(),
                  })}
                </div>
              )}
              {/* A flat line of zeros is a line that says nothing. */}
              {data.totals.runs > 0 && (
                <div className="mt-2">
                  <Sparkline
                    values={runSeries[0].values}
                    label={t("metrics.reviewedTrend")}
                  />
                </div>
              )}
            </Tile>
            <Tile
              label={t("metrics.findings")}
              value={data.totals.findings_posted.toLocaleString()}
            >
              {hasTrend(findingsDaily) && (
                <div className="mt-2">
                  <Sparkline
                    values={findingsDaily}
                    label={t("metrics.findingsTrend")}
                  />
                </div>
              )}
            </Tile>
            <Tile
              label={t("metrics.unpriced")}
              value={data.totals.unpriced_runs}
              note={t("metrics.runsNote")}
            >
              {hasTrend(unpricedDaily) && (
                <div className="mt-2">
                  <Sparkline
                    values={unpricedDaily}
                    label={t("metrics.unpricedTrend")}
                  />
                </div>
              )}
            </Tile>
          </Tiles>

          <p className={eyebrow}>{t("metrics.costPerDay")}</p>
          <Card title={t("metrics.spend")}>
            {costTop === 0 ? (
              <Empty>{t("metrics.noPriced")}</Empty>
            ) : (
              <>
                <Chart
                  series={costSeries}
                  trend={costTrend}
                  width={1290}
                  height={200}
                  mark="line"
                  ticks={costTicks}
                  xLabels={window.map(label)}
                  format={moneyTick(costTicks)}
                  label={t("metrics.costChart", { title })}
                />
                <Legend series={costSeries} trend={trendColor} />
              </>
            )}
          </Card>

          <p className={eyebrow}>{t("metrics.runsPerDay")}</p>
          <Card title={t("metrics.throughput")}>
            {runTop === 0 ? (
              <Empty>{t("metrics.noRuns")}</Empty>
            ) : (
              <>
                <Chart
                  series={runSeries}
                  trend={runTrend}
                  width={1290}
                  height={180}
                  mark="bar"
                  ticks={runTicks}
                  xLabels={window.map(label)}
                  label={t("metrics.runsChart", { title })}
                />
                <Legend series={runSeries} trend={trendColor} />
              </>
            )}
          </Card>

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
                      <th className={thNum}>{t("metrics.tokensIn")}</th>
                      <th className={thNum}>{t("metrics.tokensOut")}</th>
                      <th className={thNum}>{t("metrics.findings")}</th>
                      <th className={thNum}>{t("metrics.cost")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.by_team.map((row) => (
                      // Ranked by spend, deliberately: this table answers
                      // "who is spending", so it is not sorted by name.
                      <tr key={row.team}>
                        <td className={td} title={row.team}>
                          {row.team_name}
                        </td>
                        <td className={tdNum}>{row.runs}</td>
                        <td className={tdNum}>{tokens(row.prompt_tokens)}</td>
                        <td className={tdNum}>
                          {tokens(row.completion_tokens)}
                        </td>
                        <td className={tdNum}>{row.findings_posted}</td>
                        <td className={tdNum}>{money(row.cost_usd)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

/** A legend is always present for two or more series, so identity is never
 *  carried by colour alone; a trend's line is always named, since a lone
 *  series with a second line beside it is two things to tell apart. */
function Legend({ series, trend }: { series: Series[]; trend?: string }) {
  const { t } = useTranslation();
  if (series.length < 2 && !trend) return null;
  return (
    <div className="mt-2.5 flex flex-wrap gap-4 text-[12.5px] text-muted">
      {series.map((s) => (
        <span key={s.id} className="inline-flex items-center gap-1.5">
          <i
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-[2px]"
            style={{ background: s.color }}
          />
          {s.name}
        </span>
      ))}
      {trend && (
        <span className="inline-flex items-center gap-1.5">
          <i
            aria-hidden
            className="inline-block h-[2px] w-3.5 rounded-full"
            style={{ background: trend }}
          />
          {t("metrics.trend")}
        </span>
      )}
    </div>
  );
}

export { days as _days, niceTicks as _niceTicks };
