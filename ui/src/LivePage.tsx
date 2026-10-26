import { useTranslation } from "react-i18next";
import type { Live } from "./api";
import {
  ago,
  byLastRun,
  elapsedSince,
  outcomeTone,
  outcomeWord,
  outcomeTitle,
} from "./format";
import { usePoll, useNow } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Pill,
  PrTag,
  Sep,
  TeamPill,
  column,
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

/** Slots draws the inference pool as what it is: N slots, some busy. The
 *  pool is six wide by default, so a single "currently reviewing" row would
 *  misreport the instance as doing one thing at a time. */
function Slots({ capacity, busy }: { capacity: number; busy: number }) {
  const { t } = useTranslation();
  return (
    <div
      className="mb-3.5 flex gap-1.5"
      aria-label={t("live.slotsBusy", { busy, capacity })}
    >
      {Array.from({ length: capacity }, (_, i) => (
        <div
          key={i}
          className={
            "h-6.5 flex-1 rounded-[5px] border " +
            (i < busy
              ? "border-accent bg-accent-dim"
              : "border-border bg-panel")
          }
        />
      ))}
    </div>
  );
}

export function LivePage() {
  const { t } = useTranslation();
  // 3s: fast enough that a review starting is visible while you watch,
  // slow enough that the access log stays quiet (these paths are silenced
  // server-side for exactly this reason).
  const { data, failed, updatedAt } = usePoll<Live>("live", 3000);
  const wall = useNow(1000);
  // A failing refresh freezes the clocks at the last reading: counting on
  // against frozen rows showed a dead instance as a review stuck for ages.
  const now = failed && updatedAt !== null ? updatedAt : wall;

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>{t("nav.live")}</h2>
          <p className={lede}>{t("live.lede")}</p>
          {failed ? (
            <Failed what={t("live.failedWhat")} />
          ) : (
            <div className="skeleton h-24 rounded-ui" />
          )}
        </div>
      </div>
    );
  }

  const busy = Math.min(data.staged, data.pool_capacity);

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>{t("nav.live")}</h2>
        <p className={lede}>{t("live.lede")}</p>

        {failed && updatedAt !== null && (
          <div className="mb-3.5 rounded-ui border border-border bg-panel px-3.5 py-3 text-[13.5px] text-muted">
            {t("live.refreshFailing", {
              ago: ago(new Date(updatedAt).toISOString(), wall),
            })}
          </div>
        )}

        <Card
          title={t("live.pool")}
          // "max N per team", not "N per team": the per-team figure is a
          // ceiling nested inside the global one, not a reservation. A team
          // is not owed two slots, it is stopped from taking more than two.
          note={
            <>
              {t("live.slotsBusy", { busy, capacity: data.pool_capacity })}
              <Sep />
              {t("live.maxPerTeam", { count: data.pool_per_team })}
            </>
          }
        >
          <Slots capacity={data.pool_capacity} busy={busy} />
          {data.running.length === 0 ? (
            <Empty>{t("live.nothingRunning")}</Empty>
          ) : (
            <table className={table}>
              <tbody>
                {data.running.map((it) => (
                  <tr key={it.tag}>
                    <td className={tdTag}>
                      <PrTag tag={it.tag} base={data.bitbucket_url} />
                    </td>
                    <td className={td + " w-px"}>
                      <TeamPill name={it.team_name} slug={it.team} />
                    </td>
                    <td className={tdNum}>{elapsedSince(it.since, now)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <Card
          title={t("live.queue")}
          note={
            <>
              {t("live.waitingCount", { count: data.depth })}
              <Sep />
              {t("live.inFlightExcluded")}
            </>
          }
        >
          {data.waiting.length === 0 ? (
            <Empty>{t("live.queueEmpty")}</Empty>
          ) : (
            <table className={table}>
              <thead>
                <tr>
                  <th className={th}>{t("live.pullRequest")}</th>
                  <th className={th + " w-px"}>{t("live.team")}</th>
                  <th className={thNum}>{t("live.waiting")}</th>
                </tr>
              </thead>
              <tbody>
                {data.waiting.map((it) => (
                  <tr key={it.tag}>
                    <td className={tdTag}>
                      <PrTag tag={it.tag} base={data.bitbucket_url} />
                    </td>
                    <td className={td + " w-px"}>
                      <TeamPill name={it.team_name} slug={it.team} />
                    </td>
                    <td className={tdNum}>{elapsedSince(it.since, now)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        {/* Enabled teams that have run, and nothing else: the server filters.
            No State column, it is configuration, not live; the Teams page
            has it. The outcome pill sits in its own column, never after a
            time: a pill trailing "1m ago" reads as a label on the time. */}
        <Card title={t("live.teams")}>
          {data.teams.length === 0 ? (
            <Empty>{t("live.noRuns")}</Empty>
          ) : (
            <table className={table}>
              <thead>
                <tr>
                  <th className={th}>{t("live.team")}</th>
                  {/* nowrap: w-px shrinks the column to its widest word. */}
                  <th className={th + " w-px whitespace-nowrap"}>
                    {t("live.lastOutcome")}
                  </th>
                  <th className={thNum}>{t("live.lastReviewed")}</th>
                  <th className={thNum}>{t("live.lastRun")}</th>
                </tr>
              </thead>
              <tbody>
                {byLastRun(data.teams).map((tm) => (
                  <tr key={tm.slug}>
                    <td className={td} title={tm.slug}>
                      {tm.name}
                    </td>
                    <td className={td + " w-px"}>
                      <Pill
                        tone={outcomeTone(tm.last_outcome)}
                        title={outcomeTitle(
                          tm.last_outcome,
                          tm.last_reason,
                          tm.last_reason_label,
                        )}
                      >
                        {outcomeWord(tm.last_outcome, tm.last_reason)}
                      </Pill>
                    </td>
                    <td className={tdNum}>{ago(tm.last_reviewed, now)}</td>
                    <td className={tdNum}>{ago(tm.last_run, now)}</td>
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
