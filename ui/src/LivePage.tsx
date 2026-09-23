import type { Live } from "./api";
import { ago, byLastRun, elapsedSince, outcomeTone, outcomeWord } from "./format";
import { usePoll, useNow } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Note,
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
  return (
    <div className="mb-3.5 flex gap-1.5" aria-label={`${busy} of ${capacity} slots busy`}>
      {Array.from({ length: capacity }, (_, i) => (
        <div
          key={i}
          className={
            "h-6.5 flex-1 rounded-[5px] border " +
            (i < busy ? "border-accent bg-accent-dim" : "border-border bg-panel")
          }
        />
      ))}
    </div>
  );
}

export function LivePage() {
  // 3s: fast enough that a review starting is visible while you watch,
  // slow enough that the access log stays quiet (these paths are silenced
  // server-side for exactly this reason).
  const { data, failed } = usePoll<Live>("live", 3000);
  const now = useNow(1000);

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>Live</h2>
          <p className={lede}>What is running right now.</p>
          {failed ? <Failed what="the live panel" /> : <div className="skeleton h-24 rounded-ui" />}
        </div>
      </div>
    );
  }

  const busy = Math.min(data.staged, data.pool_capacity);

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>Live</h2>
        <p className={lede}>
          What is running right now. Several reviews can be waiting on the model at once, but only
          one at a time talks to Bitbucket, so a busy queue drains in order.
        </p>

        <Card
          title="Inference pool"
          // "max N per team", not "N per team": the per-team figure is a
          // ceiling nested inside the global one, not a reservation. A team
          // is not owed two slots, it is stopped from taking more than two.
          note={
            <>
              {busy} of {data.pool_capacity} slots busy
              <Sep />
              max {data.pool_per_team} per team
            </>
          }
        >
          <Slots capacity={data.pool_capacity} busy={busy} />
          {data.running.length === 0 ? (
            <Empty>Nothing running.</Empty>
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
          title="Queue"
          note={
            <>
              {data.depth} waiting
              <Sep />
              in-flight excluded
            </>
          }
        >
          {data.waiting.length === 0 ? (
            <Empty>Queue is empty.</Empty>
          ) : (
            <table className={table}>
              <thead>
                <tr>
                  <th className={th}>Pull request</th>
                  <th className={th + " w-px"}>Team</th>
                  <th className={thNum}>Waiting</th>
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
        <Card title="Teams">
          {data.teams.length === 0 ? (
            <Empty>No runs yet.</Empty>
          ) : (
            <table className={table}>
              <thead>
                <tr>
                  <th className={th}>Team</th>
                  {/* nowrap: w-px shrinks the column to its widest word. */}
                  <th className={th + " w-px whitespace-nowrap"}>Last outcome</th>
                  <th className={thNum}>PRs</th>
                  <th className={thNum}>Last reviewed</th>
                  <th className={thNum}>Last run</th>
                </tr>
              </thead>
              <tbody>
                {byLastRun(data.teams).map((t) => (
                  <tr key={t.slug}>
                    <td className={td} title={t.slug}>
                      {t.name}
                    </td>
                    <td className={td + " w-px"}>
                      <Pill
                        tone={outcomeTone(t.last_outcome)}
                        title={
                          t.last_outcome !== "ok" ? t.last_reason_label || t.last_reason : undefined
                        }
                      >
                        {outcomeWord(t.last_outcome, t.last_reason)}
                      </Pill>
                    </td>
                    <td className={tdNum}>{t.prs}</td>
                    <td className={tdNum}>{ago(t.last_reviewed, now)}</td>
                    <td className={tdNum}>{ago(t.last_run, now)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <Note>
            Last run counts every outcome, skipped and failed included. Last reviewed is the last
            one that succeeded.
          </Note>
        </Card>
      </div>
    </div>
  );
}
