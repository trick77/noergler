import type { Metrics, Runs } from "./api";
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
const FILTERS = [
  { value: "", label: "runs.filterAll" },
  { value: "failed", label: "runs.filterFailed" },
  { value: "skipped", label: "runs.filterSkipped" },
] as const satisfies readonly { value: Filter; label: string }[];

export function RunsPage() {
  const { t } = useTranslation();
  const [filter, setFilter] = useState<Filter>("");
  const query = `runs?limit=50${filter === "" ? "" : `&outcome=${filter}`}`;
  const { data, failed, stale } = usePoll<Runs>(query, 10000);
  // Same cadence as the feed: read once, the tiles drifted from the rows
  // beneath them for as long as the page stayed open.
  const metrics = usePoll<Metrics>("metrics", 10000);
  const now = useNow(10000);

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>{t("nav.runs")}</h2>
          {failed ? (
            <Failed what={t("runs.failedWhat")} />
          ) : (
            <div className="skeleton h-24 rounded-ui" />
          )}
        </div>
      </div>
    );
  }

  // A failing refresh dashes the counts too: kept on screen, the last
  // reading read as current while the feed beneath moved on.
  const counted = metrics.data !== null && !metrics.failed;
  const c = counts(metrics.data);
  // The skip breakdown, ranked. Only skips carry a reason; an outcome that
  // is not a skip is already named by its own word.
  const skips = (metrics.data?.breakdown ?? [])
    .filter((b) => b.outcome === "skipped")
    .sort((a, b) => b.count - a.count);

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>{t("nav.runs")}</h2>
        <p className={lede}>{t("runs.lede")}</p>

        {/* A dash, not a zero, until the counts are read, failed or still
            loading: the feed below may be full, and four tiles reading 0
            beside it would be a claim rather than an absence. */}
        <Tiles>
          <Tile label={t("runs.tileRuns")} value={counted ? c.total : "—"} />
          <Tile
            label={t("runs.tileReviewed")}
            value={counted ? c.reviewed : "—"}
          />
          <Tile
            label={t("runs.tileSkipped")}
            value={counted ? c.skipped : "—"}
          />
          <Tile label={t("runs.tileFailed")} value={counted ? c.failed : "—"} />
        </Tiles>

        <p className={eyebrow}>{t("runs.latest")}</p>
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
                  {t(f.label)}
                </button>
              ))}
            </span>
          }
        >
          {data.runs.length === 0 ? (
            <Empty>
              {filter === ""
                ? t("runs.empty")
                : filter === "failed"
                  ? t("runs.emptyFailed")
                  : t("runs.emptySkipped")}
            </Empty>
          ) : (
            <div
              className={
                "overflow-x-auto overscroll-x-contain transition-opacity " +
                // Still the previous filter's rows: shown on their way out
                // rather than presented as the answer to what was just asked.
                (stale ? "opacity-40" : "")
              }
            >
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
                  {data.runs.map((r, i) => (
                    <tr key={`${r.tag}-${r.created_at}-${i}`}>
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
          {metrics.failed ? (
            <Empty>{t("runs.countsUnavailable")}</Empty>
          ) : metrics.data === null ? (
            <div className="skeleton h-12 rounded-ui" />
          ) : skips.length === 0 ? (
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
