import type { Team } from "./api";
import { ago, scope, teamStateLabel, teamTone } from "./format";
import { usePoll, useNow } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Note,
  Pill,
  column,
  h2,
  lede,
  page,
  table,
  tdNum,
  tdTag,
  th,
  thNum,
} from "./ui";

function List({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <>
      {" "}
      {label}: <code className="font-mono text-[12px] text-muted">{items.join(", ")}</code>
    </>
  );
}

/** Count renders an author list's size. The names are not served here: this
 *  page is unauthenticated, and a roster of who works where is not the same
 *  question as "is this team configured". */
function Count({ label, n }: { label: string; n: number }) {
  if (n === 0) return null;
  return (
    <>
      {" "}
      {label}: <span className="text-muted">{n}</span>
    </>
  );
}

export function TeamsPage() {
  const { data, failed } = usePoll<{ teams: Team[] }>("teams", 30000);
  const now = useNow(30000);

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>Teams</h2>
          {failed ? <Failed what="the team roster" /> : <div className="skeleton h-24 rounded-ui" />}
        </div>
      </div>
    );
  }

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>Teams</h2>
        <p className={lede}>
          Read-only. A team is a webhook path, its own HMAC secret and the projects and repositories
          it has claimed.
        </p>

        {data.teams.map((t) => (
          <Card
            key={t.slug}
            title={t.slug}
            note={`${t.enabled ? scope(t.repos) : "not started"} · ${t.prs} PRs · last run ${ago(t.last_run, now)}`}
            right={<Pill tone={teamTone(t.state)}>{teamStateLabel(t.state)}</Pill>}
          >
            {t.claims.length === 0 ? (
              <Empty>
                No projects and no repositories claimed, so this team reviews nothing.
              </Empty>
            ) : (
              <table className={table}>
                <thead>
                  <tr>
                    <th className={th}>Claim</th>
                    <th className={thNum}>Scope</th>
                  </tr>
                </thead>
                <tbody>
                  {t.claims.map((c) => (
                    <tr key={`${c.project}/${c.repo ?? ""}`}>
                      <td className={tdTag}>{c.repo ? `${c.project}/${c.repo}` : c.project}</td>
                      <td className={tdNum}>{c.repo ? "repo" : "whole project"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {(t.exclude_repos.length > 0 ||
              t.ignore_authors > 0 ||
              t.auto_review_authors > 0) && (
              <Note>
                <List label="Excluded repos" items={t.exclude_repos} />
                <Count label="Auto-review authors" n={t.auto_review_authors} />
                <Count label="Ignored authors" n={t.ignore_authors} />
              </Note>
            )}
            {!t.enabled && (
              <Note>
                The reason a team is disabled is not shown here and not served by the API: it can
                name an environment variable. It is in the pod log, on the{" "}
                <code className="font-mono text-[12px]">team_disabled</code> line.
              </Note>
            )}
          </Card>
        ))}
        {data.teams.length === 0 && <Empty>No teams configured.</Empty>}
      </div>
    </div>
  );
}
