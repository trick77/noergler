import type { Metrics, Runs } from "./api";
import { ago, duration, money, outcomeTone } from "./format";
import { useState } from "react";
import { usePoll, useNow } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Pill,
  PrTag,
  TeamPill,
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
  tdTag,
  th,
  thNum,
} from "./ui";

interface Counts {
  total: number;
  reviewed: number;
  skipped: number;
  failed: number;
}

function counts(m: Metrics | null): Counts {
  const c: Counts = { total: 0, reviewed: 0, skipped: 0, failed: 0 };
  for (const b of m?.breakdown ?? []) {
    c.total += b.count;
    if (b.outcome === "ok") c.reviewed += b.count;
    else if (b.outcome === "skipped") c.skipped += b.count;
    else c.failed += b.count;
  }
  return c;
}

/** The outcome filter. Server-side, not over the fetched page: the feed is a
 *  window, so filtering what came back can show nothing while failures sit
 *  just past its edge - which reads as "nothing failed". */
type Filter = "" | "failed" | "skipped";
const FILTERS: { value: Filter; label: string }[] = [
  { value: "", label: "All" },
  { value: "failed", label: "Failed" },
  { value: "skipped", label: "Skipped" },
];

export function RunsPage() {
  const [filter, setFilter] = useState<Filter>("");
  const query = `runs?limit=50${filter === "" ? "" : `&outcome=${filter}`}`;
  const { data, failed } = usePoll<Runs>(query, 10000);
  const metrics = usePoll<Metrics>("metrics");
  const now = useNow(10000);

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>Runs</h2>
          {failed ? <Failed what="the runs feed" /> : <div className="skeleton h-24 rounded-ui" />}
        </div>
      </div>
    );
  }

  const c = counts(metrics.data);
  // The skip breakdown, ranked. Only skips carry a reason; an outcome that
  // is not a skip is already named by its own word.
  const skips = (metrics.data?.breakdown ?? [])
    .filter((b) => b.outcome === "skipped")
    .sort((a, b) => b.count - a.count);

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>Runs</h2>
        <p className={lede}>
          Every PR noergler looked at this month, including the ones it decided not to review.
          Skipped means it never asked the model. Failed means it asked and something went wrong.
        </p>

        {/* A dash, not a zero, when the counts could not be read: the feed
            below may be full, and four tiles reading 0 beside it would be a
            claim rather than an absence. */}
        <Tiles>
          <Tile label="Attempts" value={metrics.failed ? "—" : c.total} />
          <Tile label="Reviewed" value={metrics.failed ? "—" : c.reviewed} />
          <Tile label="Skipped" value={metrics.failed ? "—" : c.skipped} />
          <Tile label="Failed" value={metrics.failed ? "—" : c.failed} />
        </Tiles>

        <p className={eyebrow}>Recent attempts</p>
        <Card
          right={
            <span className="flex gap-0.5">
              {FILTERS.map((f) => (
                <button
                  key={f.value}
                  type="button"
                  onClick={() => setFilter(f.value)}
                  aria-pressed={filter === f.value}
                  className={
                    "rounded-ui-sm px-2 py-0.5 text-[12px] transition-colors " +
                    (filter === f.value
                      ? "bg-panel text-ink"
                      : "text-muted hover:bg-panel hover:text-ink")
                  }
                >
                  {f.label}
                </button>
              ))}
            </span>
          }
        >
          {data.runs.length === 0 ? (
            <Empty>
              {filter === "" ? "No attempts recorded yet." : `No ${filter} attempts recorded yet.`}
            </Empty>
          ) : (
            <div className="overflow-x-auto overscroll-x-contain">
              <table className={table}>
                <thead>
                  <tr>
                    <th className={th}>Pull request</th>
                    <th className={th + " w-px"}>Team</th>
                    <th className={th + " w-px"}>Outcome</th>
                    <th className={thNum}>Findings</th>
                    <th className={thNum}>Elapsed</th>
                    <th className={thNum}>Cost</th>
                    <th className={thNum}>When</th>
                  </tr>
                </thead>
                <tbody>
                  {data.runs.map((r, i) => (
                    <tr key={`${r.tag}-${r.created_at}-${i}`}>
                      <td className={tdTag}>
                        <PrTag tag={r.tag} base={data.bitbucket_url} />
                      </td>
                      <td className={td + " w-px"}>
                        <TeamPill name={r.team_name} slug={r.team} />
                      </td>
                      <td className={td}>
                        <Pill tone={outcomeTone(r.outcome)}>
                          {r.outcome === "skipped" ? "skipped" : r.outcome}
                        </Pill>
                        {/* A red pill that only says "failed" sends the
                            reader to the logs for something the row already
                            knows. */}
                        {r.outcome !== "ok" && (r.reason_label || r.reason) && (
                          <span className="ml-2 text-[12px] text-faint">
                            {r.reason_label || r.reason}
                          </span>
                        )}
                      </td>
                      <td className={tdNum}>{r.findings ?? "—"}</td>
                      <td className={tdNum}>{duration(r.elapsed_ms)}</td>
                      <td
                        className={
                          tdNum + (r.cost_usd === null && r.outcome === "ok" ? " text-faint italic" : "")
                        }
                      >
                        {r.outcome === "ok" ? money(r.cost_usd) : "—"}
                      </td>
                      <td className={tdNum + " text-faint"}>{ago(r.created_at, now)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <p className={eyebrow}>Why runs were skipped</p>
        <Card>
          {metrics.failed ? (
            <Empty>Counts unavailable.</Empty>
          ) : skips.length === 0 ? (
            <Empty>Nothing was skipped.</Empty>
          ) : (
            <table className={table}>
              <thead>
                <tr>
                  <th className={th}>Reason</th>
                  <th className={thNum}>Count</th>
                </tr>
              </thead>
              <tbody>
                {skips.map((b) => (
                  <tr key={b.reason}>
                    <td className={td}>{b.label ?? b.reason}</td>
                    <td className={tdNum}>{b.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </div>
    </div>
  );
}
