import type { Live } from "./api";
import { ago, byName, elapsedSince, teamStateLabel, teamTone } from "./format";
import { usePoll, useNow } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Note,
  Pill,
  PrTag,
  SEP,
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
          What is running right now. The inference pool overlaps the gateway calls; every Bitbucket
          fetch and every comment posted still happens one at a time on the single review worker.
        </p>

        <Card
          title="Inference pool"
          // "max N per team", not "N per team": the per-team figure is a
          // ceiling nested inside the global one, not a reservation. A team
          // is not owed two slots, it is stopped from taking more than two.
          note={`${busy} of ${data.pool_capacity} slots busy${SEP}max ${data.pool_per_team} per team`}
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

        <Card title="Queue" note={`${data.depth} waiting${SEP}in-flight excluded`}>
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

        <Card title="Teams">
          <table className={table}>
            <thead>
              <tr>
                <th className={th}>Team</th>
                <th className={th + " w-px"}>State</th>
                <th className={thNum}>PRs</th>
                <th className={thNum}>Last run</th>
              </tr>
            </thead>
            <tbody>
              {byName(data.teams).map((t) => (
                <tr key={t.slug}>
                  <td className={td} title={t.slug}>
                    {t.name}
                  </td>
                  <td className={td + " w-px"}>
                    <Pill tone={teamTone(t.state)}>{teamStateLabel(t.state)}</Pill>
                  </td>
                  <td className={tdNum}>{t.prs}</td>
                  <td className={tdNum}>{ago(t.last_run, now)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Note>
            A team with no repositories is configured and running and reviewing nothing: it owns no
            project and no repo. Why a team is disabled stays in the logs, because the reason can
            name an env var.
          </Note>
        </Card>
      </div>
    </div>
  );
}
