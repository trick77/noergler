import type { Metrics } from "./api";
import { Chart, Sparkline, type Series } from "./Chart";
import { money, tokens } from "./format";
import { usePoll } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Note,
  Tile,
  Tiles,
  column,
  eyebrow,
  h2,
  lede,
  page,
  table,
  tdNum,
  tdTag,
  th,
  thNum,
} from "./ui";

// Fixed order, never cycled: a team keeps its colour as teams come and go,
// so a filter that drops one does not repaint the others.
const SERIES_COLORS = ["var(--color-s1)", "var(--color-s2)", "var(--color-s3)", "var(--color-s4)"];

/** days lists every date in [since, until), so a day with no runs is a zero
 *  rather than a gap the chart closes over. The API returns only days that
 *  have rows, deliberately: filling them is the caller's job, and only the
 *  window itself knows how wide it is. A month is 28 to 31 days, so this is
 *  never a fixed count.
 *
 *  The window is the SERVER's calendar month, so its ends are read off the
 *  ISO string rather than through the browser's clock. `new Date(iso)` plus
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
  // A month is at most 31 days; the bound stops a bad pair spinning.
  for (let i = 0; d.getTime() < end && i < 400; i++) {
    out.push(d.toISOString().slice(0, 10));
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return out;
}

/** calendarDay reads the Y-M-D the server wrote, as a UTC timestamp, with no
 *  reference to the viewer's zone. Returns null for anything unparseable, so
 *  a bad pair renders an empty window rather than 400 wrong days. */
function calendarDay(iso: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (m === null) return null;
  return Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
}

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** windowTitle names the window the API actually answered with, rather than
 *  the one the page assumed it asked for. */
function windowTitle(m: Metrics): string {
  if (m.window !== "month") return `the last ${days(m.since, m.until).length} days`;
  // The server's month, read off the string: through the viewer's clock a
  // +02:00 server's September titles itself "August" in New York.
  const start = calendarDay(m.since);
  if (start === null) return "this month";
  const d = new Date(start);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
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
  // No ?days=: the default window is the current calendar month, which is
  // the period a key's spend is budgeted and invoiced in.
  const { data, failed } = usePoll<Metrics>("metrics", 60000);

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>Metrics</h2>
          {failed ? <Failed what="metrics" /> : <div className="skeleton h-24 rounded-ui" />}
        </div>
      </div>
    );
  }

  const window = days(data.since, data.until);
  const title = windowTitle(data);
  // Day-of-month labels, every third, so they never collide. A month reads
  // as dates rather than as an offset from today: "d-13" is meaningless
  // when the window has a fixed start.
  const step = window.length > 20 ? 4 : 3;
  const label = (day: string, i: number) =>
    i % step === 0 ? String(Number(day.slice(-2))) : "";

  // Cost per day per team. A team's missing day is 0 for the chart's
  // purposes, which is true: it spent nothing.
  //
  // Team order comes from by_team, which the server sorts by spend. Taking
  // it from data.daily instead ordered by whichever team happened to appear
  // first on the earliest day, which is unspecified within a day: the same
  // data could repaint every series a different colour between two polls.
  const ranked = data.by_team.map((t) => t.team);
  const charted = ranked.slice(0, SERIES_COLORS.length - 1);
  const rest = ranked.slice(SERIES_COLORS.length - 1);

  const spendOn = (day: string, team: string) => {
    const row = data.daily.find((b) => b.day === day && b.team === team);
    return row?.cost_usd ? Number(row.cost_usd) : 0;
  };

  const costSeries: Series[] = charted.map((team, i) => ({
    name: team,
    color: SERIES_COLORS[i],
    values: window.map((day) => spendOn(day, team)),
  }));
  // The tail folds into one band rather than being dropped. Truncating it
  // let the card say "No priced runs in this window" beside a Cost tile
  // showing the server's full, non-zero total.
  if (rest.length > 0) {
    costSeries.push({
      name: rest.length === 1 ? rest[0] : `${rest.length} more`,
      color: SERIES_COLORS[SERIES_COLORS.length - 1],
      values: window.map((day) => rest.reduce((sum, team) => sum + spendOn(day, team), 0)),
    });
  }
  const costTop = Math.max(
    ...window.map((_, i) => costSeries.reduce((sum, s) => sum + s.values[i], 0)),
    0,
  );

  // Attempts per day, by outcome. Skipped and failed use the status hues,
  // not the series ramp: they are states, not categories.
  const bucket = (test: (o: string) => boolean) =>
    window.map((day) =>
      data.daily_attempts
        .filter((b) => b.day === day && test(b.outcome))
        .reduce((n, b) => n + b.count, 0),
    );
  const runSeries: Series[] = [
    { name: "reviewed", color: "var(--color-s1)", values: bucket((o) => o === "ok") },
    { name: "skipped", color: "var(--color-ochre)", values: bucket((o) => o === "skipped") },
    {
      name: "failed",
      color: "var(--color-danger)",
      values: bucket((o) => o !== "ok" && o !== "skipped"),
    },
  ];
  const runTop = Math.max(
    ...window.map((_, i) => runSeries.reduce((sum, s) => sum + s.values[i], 0)),
    0,
  );

  const costTotals = costSeries.length
    ? window.map((_, i) => costSeries.reduce((sum, s) => sum + s.values[i], 0))
    : [];

  return (
    <div className={page}>
      <div className={column}>
        {/* The window is named ONCE, here. Repeating it on every tile and
            card says the same thing five times and crowds out what each
            card actually measures. */}
        <h2 className={h2}>
          Metrics <span className="text-muted">· {title}</span>
        </h2>
        <p className={lede}>
          Spend and throughput for the current month, the period a key's spend is budgeted in. Cost
          is summed from priced runs only; an unpriced run is counted separately and never folded in
          as zero.
        </p>

        <Tiles>
          {/* On an instance that has not run yet, "unpriced" would claim the
              gateway declined to price something. Nothing ran; that is a
              dash. "unpriced" is reserved for runs that actually happened
              and came back without a price. */}
          <Tile label="Cost" value={data.totals.runs === 0 ? "—" : money(data.totals.cost_usd)}>
            {costTop > 0 && (
              <div className="mt-2">
                <Sparkline values={costTotals} label="Cost trend" />
              </div>
            )}
          </Tile>
          <Tile label="Runs" value={data.totals.runs.toLocaleString()}>
            {/* A flat line of zeros is a line that says nothing. */}
            {data.totals.runs > 0 && (
              <div className="mt-2">
                <Sparkline values={runSeries[0].values} label="Run count trend" />
              </div>
            )}
          </Tile>
          <Tile label="Findings" value={data.totals.findings_posted.toLocaleString()} />
          <Tile label="Unpriced" value={data.totals.unpriced_runs} note="runs" />
        </Tiles>

        <p className={eyebrow}>Cost per day</p>
        <Card title="Spend" note="USD, priced runs only">
          {costTop === 0 ? (
            <Empty>No priced runs in this window.</Empty>
          ) : (
            <>
              <Chart
                series={costSeries}
                width={856}
                height={200}
                mark="stack"
                ticks={niceTicks(costTop)}
                xLabels={window.map(label)}
                format={(v) => `$${v.toFixed(2)}`}
                label={`Cost per day for ${title}, by team`}
              />
              <Legend series={costSeries} />
            </>
          )}
        </Card>

        <p className={eyebrow}>Runs per day</p>
        <Card title="Throughput" note="reviewed, skipped and failed attempts">
          {runTop === 0 ? (
            <Empty>No attempts in this window.</Empty>
          ) : (
            <>
              <Chart
                series={runSeries}
                width={856}
                height={180}
                mark="bar"
                ticks={niceTicks(runTop)}
                xLabels={window.map(label)}
                label={`Attempts per day for ${title}`}
              />
              <Legend series={runSeries} />
            </>
          )}
        </Card>

        <p className={eyebrow}>By team</p>
        <Card>
          {data.by_team.length === 0 ? (
            <Empty>No runs in this window.</Empty>
          ) : (
            <div className="overflow-x-auto overscroll-x-contain">
              <table className={table}>
                <thead>
                  <tr>
                    <th className={th}>Team</th>
                    <th className={thNum}>Runs</th>
                    <th className={thNum}>Tokens in</th>
                    <th className={thNum}>Tokens out</th>
                    <th className={thNum}>Findings</th>
                    <th className={thNum}>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {data.by_team.map((t) => (
                    <tr key={t.team}>
                      <td className={tdTag}>{t.team}</td>
                      <td className={tdNum}>{t.runs}</td>
                      <td className={tdNum}>{tokens(t.prompt_tokens)}</td>
                      <td className={tdNum}>{tokens(t.completion_tokens)}</td>
                      <td className={tdNum}>{t.findings_posted}</td>
                      <td className={tdNum}>{money(t.cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <Note>Key spend is a gauge: shown, never summed across teams.</Note>
        </Card>
      </div>
    </div>
  );
}

/** A legend is always present for two or more series, so identity is never
 *  carried by colour alone. */
function Legend({ series }: { series: Series[] }) {
  if (series.length < 2) return null;
  return (
    <div className="mt-2.5 flex flex-wrap gap-4 text-[12.5px] text-muted">
      {series.map((s) => (
        <span key={s.name} className="inline-flex items-center gap-1.5">
          <i
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-[2px]"
            style={{ background: s.color }}
          />
          {s.name}
        </span>
      ))}
    </div>
  );
}

export { days as _days, niceTicks as _niceTicks, windowTitle as _windowTitle };
