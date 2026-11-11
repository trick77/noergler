import type { Run, Runs } from "./api";
import {
  ago,
  duration,
  money,
  outcomeTone,
  outcomeTitle,
  outcomeWord,
  reasonLabel,
} from "./format";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { RUNS_LIMIT } from "./runsLimit";
import { usePoll, useNow } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Pill,
  PrTag,
  FilterTrack,
  RefreshFailing,
  TeamPill,
  Tile,
  Tiles,
  column,
  eyebrow,
  page,
  pageTitle,
  table,
  td,
  tdNum,
  tdTag,
  th,
  thNum,
} from "./ui";

// The page is the latest RUNS_LIMIT runs, nothing wider (user, 2026-11-10):
// tiles, table and skip breakdown all read the same rows, so none of them
// can disagree with another or load on its own. No "Runs" tile: it would
// read RUNS_LIMIT on any instance that has run that often.

interface Counts {
  reviewed: number;
  skipped: number;
  failed: number;
}

function counts(runs: Run[]): Counts {
  const c: Counts = { reviewed: 0, skipped: 0, failed: 0 };
  for (const r of runs) {
    if (r.outcome === "ok") c.reviewed++;
    else if (r.outcome === "skipped") c.skipped++;
    else c.failed++;
  }
  return c;
}

/** skipReasons is the skip breakdown, ranked. Only skips carry a reason; an
 *  outcome that is not a skip is already named by its own word. */
function skipReasons(runs: Run[]) {
  const by = new Map<
    string,
    { reason: string; label?: string; count: number }
  >();
  for (const r of runs) {
    if (r.outcome !== "skipped") continue;
    const key = r.reason ?? "";
    const b = by.get(key) ?? { reason: key, label: r.reason_label, count: 0 };
    b.count++;
    by.set(key, b);
  }
  // Ties by label: insertion order is newest run first, so equal counts
  // swapped places as the feed moved and the table jumped every poll.
  const label = (b: { reason: string; label?: string }) =>
    reasonLabel(b.reason, b.label) ?? "";
  return [...by.values()].sort(
    (a, b) => b.count - a.count || label(a).localeCompare(label(b)),
  );
}

/** The outcome filter narrows the same rows the tiles count, in the
 *  browser: "Failed 2" on a tile is the two rows the filter shows. A failure
 *  older than the latest runs is the API's (?outcome=failed), not the
 *  page's. */
type Filter = "" | "failed" | "skipped";
const FILTERS = [
  { value: "", label: "runs.filterAll" },
  { value: "failed", label: "runs.filterFailed" },
  { value: "skipped", label: "runs.filterSkipped" },
] as const satisfies readonly { value: Filter; label: string }[];

function keep(filter: Filter, r: Run): boolean {
  if (filter === "skipped") return r.outcome === "skipped";
  if (filter === "failed") return r.outcome !== "ok" && r.outcome !== "skipped";
  return true;
}

export function RunsPage() {
  const { t } = useTranslation();
  const [filter, setFilter] = useState<Filter>("");
  const { data, failed, updatedAt } = usePoll<Runs>(
    `runs?limit=${RUNS_LIMIT}`,
    10000,
  );
  const wall = useNow(10000);
  // A failing refresh freezes the "when" column at the last reading, as on
  // Live: ages counting on beside stale rows read as a live page.
  const now = failed && updatedAt !== null ? updatedAt : wall;

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={pageTitle}>{t("nav.runs")}</h2>
          {failed ? (
            <Failed what={t("runs.failedWhat")} />
          ) : (
            <div className="skeleton h-24 rounded-ui" />
          )}
        </div>
      </div>
    );
  }

  const c = counts(data.runs);
  const skips = skipReasons(data.runs);
  const rows = data.runs.filter((r) => keep(filter, r));

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={pageTitle}>{t("nav.runs")}</h2>

        {failed && updatedAt !== null && (
          <RefreshFailing at={updatedAt} now={wall} />
        )}

        <Tiles cols={3}>
          <Tile label={t("runs.tileReviewed")} value={c.reviewed} />
          <Tile label={t("runs.tileSkipped")} value={c.skipped} />
          <Tile label={t("runs.tileFailed")} value={c.failed} />
        </Tiles>

        <p className={eyebrow}>{t("runs.latest", { n: RUNS_LIMIT })}</p>
        <Card
          right={
            <FilterTrack
              label={t("runs.outcome")}
              options={FILTERS.map((f) => ({
                value: f.value,
                label: t(f.label),
              }))}
              value={filter}
              onChange={setFilter}
            />
          }
        >
          {rows.length === 0 ? (
            <Empty>
              {filter === ""
                ? t("runs.empty")
                : filter === "failed"
                  ? t("runs.emptyFailed", { n: RUNS_LIMIT })
                  : t("runs.emptySkipped", { n: RUNS_LIMIT })}
            </Empty>
          ) : (
            <div className="overflow-x-auto overscroll-x-contain">
              <table className={table}>
                <thead>
                  <tr>
                    <th className={th}>{t("runs.pullRequest")}</th>
                    <th className={th + " w-px"}>{t("runs.team")}</th>
                    <th className={th + " w-px"}>{t("runs.outcome")}</th>
                    <th className={thNum}>{t("runs.findings")}</th>
                    <th className={thNum}>{t("runs.elapsed")}</th>
                    <th className={thNum}>{t("runs.cost")}</th>
                    <th className={thNum}>{t("runs.when")}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.id}>
                      <td className={tdTag}>
                        <PrTag tag={r.tag} base={data.bitbucket_url} />
                      </td>
                      <td className={td + " w-px"}>
                        <TeamPill name={r.team_name} slug={r.team} />
                      </td>
                      <td className={td}>
                        {/* The reason is the pill's word, its long label the
                            tooltip: a label beside the pill wrapped the cell
                            into three or four lines. Never a link: a run's
                            page is reached only from its summary comment. */}
                        <Pill
                          tone={outcomeTone(r.outcome)}
                          title={outcomeTitle(
                            r.outcome,
                            r.reason,
                            r.reason_label,
                          )}
                        >
                          {outcomeWord(r.outcome, r.reason)}
                        </Pill>
                      </td>
                      <td className={tdNum}>{r.findings ?? "—"}</td>
                      <td className={tdNum}>{duration(r.elapsed_ms)}</td>
                      <td
                        className={
                          tdNum +
                          (r.cost_usd === null && r.outcome === "ok"
                            ? " text-faint italic"
                            : "")
                        }
                      >
                        {r.outcome === "ok" ? money(r.cost_usd) : "—"}
                      </td>
                      <td className={tdNum + " text-faint"}>
                        {ago(r.created_at, now)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <p className={eyebrow}>{t("runs.whySkipped")}</p>
        <Card>
          {skips.length === 0 ? (
            <Empty>{t("runs.nothingSkipped")}</Empty>
          ) : (
            <table className={table}>
              <thead>
                <tr>
                  <th className={th}>{t("runs.reason")}</th>
                  <th className={thNum}>{t("runs.count")}</th>
                </tr>
              </thead>
              <tbody>
                {skips.map((b) => (
                  <tr key={b.reason}>
                    <td className={td}>{reasonLabel(b.reason, b.label)}</td>
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
