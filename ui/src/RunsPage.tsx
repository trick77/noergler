import type { Run, Runs } from "./api";
import {
  actWord,
  ago,
  duration,
  money,
  outcomeTone,
  outcomeTitle,
  outcomeWord,
  reasonLabel,
  type Tone,
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

// The page is the latest RUNS_LIMIT acts, nothing wider (user, 2026-11-10):
// tiles, table and skip breakdown all read the same rows, so none of them
// can disagree with another or load on its own. No "Runs" tile: it would
// read RUNS_LIMIT on any instance that has run that often.
//
// Every act its own row (user, 2026-11-17: "eigene Einträge"): a review, an
// engagement, a fix request, a SNAPSHOT task. A review's tasks ride its row
// beside its findings, never in them.

/** The act a row is; an older server sends none, every row a review. */
const actOf = (r: Run) => r.act ?? "review";

/** Fix outcomes that went wrong, as the server's ?outcome=failed reads
 *  them: the call failed, the change was rejected as unsafe, or Bitbucket
 *  refused the commit. "Not feasible" is an answer, not a failure. */
const FIX_FAILURES = ["error", "conflict", "refused", "no_write", "rejected"];

/** isFailure is a row that went wrong: a review neither ok nor skipped, a
 *  failed engagement, a fix that went wrong. */
function isFailure(r: Run): boolean {
  switch (actOf(r)) {
    case "review":
      return r.outcome !== "ok" && r.outcome !== "skipped";
    case "engagement":
      return r.outcome === "failed";
    case "fix":
      return FIX_FAILURES.includes(r.outcome);
    default:
      return false;
  }
}

interface Counts {
  reviewed: number;
  skipped: number;
  failed: number;
  engagements: number;
  fixes: number;
}

function counts(runs: Run[]): Counts {
  const c: Counts = {
    reviewed: 0,
    skipped: 0,
    failed: 0,
    engagements: 0,
    fixes: 0,
  };
  for (const r of runs) {
    const act = actOf(r);
    if (isFailure(r)) c.failed++;
    if (act === "engagement") c.engagements++;
    else if (act === "fix") c.fixes++;
    else if (act !== "review") continue;
    else if (r.outcome === "ok") c.reviewed++;
    else if (r.outcome === "skipped") c.skipped++;
  }
  return c;
}

/** unpriced is a row the gateway billed without a price: a review that ran,
 *  or an engagement with a call it did not price (the server then sends no
 *  cost at all, never the priced part alone). */
function unpriced(r: Run): boolean {
  if (r.cost_usd !== null) return false;
  return (
    (actOf(r) === "review" && r.outcome === "ok") ||
    (actOf(r) === "engagement" && r.unpriced === true)
  );
}

/** priced is a row with a price to show. */
function priced(r: Run): boolean {
  return (
    r.cost_usd !== null &&
    ((actOf(r) === "review" && r.outcome === "ok") || actOf(r) === "engagement")
  );
}

/** The acts the type filter offers, "" every one. */
const ACTS = ["", "review", "engagement", "fix", "task"] as const;
type Act = (typeof ACTS)[number];

/** actTone is a non-review row's pill tone: done, gone wrong, held at the
 *  cost cap (a decision, like a skip), or a refusal that is a fact. */
function actTone(r: Run): Tone {
  const o = r.outcome;
  if (o === "answered" || o === "committed" || o === "posted") return "ok";
  if (isFailure(r)) return "fail";
  if (o === "refused" && actOf(r) === "engagement") return "skip";
  return "quiet";
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

function keep(filter: Filter, act: Act, r: Run): boolean {
  if (act !== "" && actOf(r) !== act) return false;
  if (filter === "skipped")
    return actOf(r) === "review" && r.outcome === "skipped";
  if (filter === "failed") return isFailure(r);
  return true;
}

export function RunsPage() {
  const { t } = useTranslation();
  const [filter, setFilter] = useState<Filter>("");
  const [act, setAct] = useState<Act>("");
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
  const rows = data.runs.filter((r) => keep(filter, act, r));
  /** actLabel names a row's act; one this build lacks prints as it came. */
  const actLabel = (a: string) =>
    a === "review" || a === "engagement" || a === "fix" || a === "task"
      ? t(`runs.actOne.${a}`)
      : a;
  /** details says what the act was about, in words: a review's findings
   *  and tasks apart, an engagement's trigger, a fix's source and commit. */
  const details = (r: Run) => {
    switch (actOf(r)) {
      case "review":
        // No run row (its write failed): the figures are missing, which is
        // a dash. A zero is left out too, never "0 findings" (user,
        // 2026-11-18): a clean review reads as a dash.
        if (r.outcome !== "ok" || r.findings === null) return "";
        return [
          ...(r.findings > 0
            ? [t("runs.detailFindings", { count: r.findings })]
            : []),
          ...((r.tasks ?? 0) > 0
            ? [t("runs.detailTasks", { count: r.tasks ?? 0 })]
            : []),
        ].join(" · ");
      // A detail this build has no word for prints as it came.
      case "engagement":
        return r.detail === "mention" || r.detail === "reply"
          ? t(`runs.trigger.${r.detail}`)
          : (r.detail ?? "");
      case "fix":
        return r.detail === "finding" ||
          r.detail === "answer" ||
          r.detail === "conflict"
          ? t(`runs.fixSource.${r.detail}`)
          : (r.detail ?? "");
      case "task":
        return t("runs.snapshot");
      default:
        return "";
    }
  };

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={pageTitle}>{t("nav.runs")}</h2>

        {failed && updatedAt !== null && (
          <RefreshFailing at={updatedAt} now={wall} />
        )}

        <Tiles cols={5}>
          <Tile label={t("runs.tileReviewed")} value={c.reviewed} />
          <Tile label={t("runs.tileSkipped")} value={c.skipped} />
          <Tile label={t("runs.tileFailed")} value={c.failed} />
          <Tile label={t("runs.tileEngagements")} value={c.engagements} />
          <Tile label={t("runs.tileFixes")} value={c.fixes} />
        </Tiles>

        <p className={eyebrow}>{t("runs.latest", { n: RUNS_LIMIT })}</p>
        <Card
          right={
            <span className="flex flex-wrap justify-end gap-2">
              <FilterTrack
                label={t("runs.act")}
                options={ACTS.map((a) => ({
                  value: a,
                  label: a === "" ? t("runs.filterAll") : t(`runs.acts.${a}`),
                }))}
                value={act}
                onChange={setAct}
              />
              <FilterTrack
                label={t("runs.outcome")}
                options={FILTERS.map((f) => ({
                  value: f.value,
                  label: t(f.label),
                }))}
                value={filter}
                onChange={setFilter}
              />
            </span>
          }
        >
          {rows.length === 0 ? (
            <Empty>
              {filter === "" && act === ""
                ? t("runs.empty")
                : filter === "failed"
                  ? t("runs.emptyFailed", { n: RUNS_LIMIT })
                  : filter === "skipped"
                    ? t("runs.emptySkipped", { n: RUNS_LIMIT })
                    : t("runs.emptyAct", { n: RUNS_LIMIT })}
            </Empty>
          ) : (
            <div className="overflow-x-auto overscroll-x-contain">
              <table className={table}>
                <thead>
                  <tr>
                    <th className={th}>{t("runs.pullRequest")}</th>
                    <th className={th + " w-px"}>{t("runs.team")}</th>
                    <th className={th + " w-px"}>{t("runs.act")}</th>
                    <th className={th + " w-px"}>{t("runs.outcome")}</th>
                    <th className={th}>{t("runs.details")}</th>
                    <th className={thNum}>{t("runs.elapsed")}</th>
                    <th className={thNum}>{t("runs.cost")}</th>
                    <th className={thNum}>{t("runs.when")}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={`${actOf(r)}:${r.id}`}>
                      <td className={tdTag}>
                        <PrTag tag={r.tag} base={data.bitbucket_url} />
                      </td>
                      <td className={td + " w-px"}>
                        <TeamPill name={r.team_name} slug={r.team} />
                      </td>
                      <td className={td + " w-px whitespace-nowrap text-muted"}>
                        {actLabel(actOf(r))}
                      </td>
                      <td className={td}>
                        {/* The reason is the pill's word, its long label the
                            tooltip: a label beside the pill wrapped the cell
                            into three or four lines. Never a link: a run's
                            page is reached only from its summary comment. */}
                        {actOf(r) === "review" ? (
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
                        ) : (
                          <Pill tone={actTone(r)}>
                            {actWord(actOf(r), r.outcome)}
                          </Pill>
                        )}
                      </td>
                      <td className={td + " whitespace-nowrap text-muted"}>
                        {details(r) || <span className="text-faint">—</span>}
                        {r.commit && (
                          <span className="ml-2 font-mono text-[12.5px]">
                            {r.commit.slice(0, 7)}
                          </span>
                        )}
                      </td>
                      <td className={tdNum}>{duration(r.elapsed_ms)}</td>
                      {/* A priced row its price; an unpriced review or
                          engagement "unpriced", never a partial sum or a
                          dash; a row with no call a dash. */}
                      <td
                        className={
                          tdNum + (unpriced(r) ? " text-faint italic" : "")
                        }
                      >
                        {unpriced(r) || priced(r) ? money(r.cost_usd) : "—"}
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
