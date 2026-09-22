import { Fragment, type ReactNode } from "react";
import type { Team } from "./api";
import { ago, byName, scope, teamStateLabel, teamTone } from "./format";
import { usePoll, useNow } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Note,
  Pill,
  Sep,
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

/** settingsParts is the team's non-empty settings, in display order.
 *
 *  The author lists are COUNTS, never names: this page is unauthenticated,
 *  and a roster of who works where is a different question from "is this
 *  team configured".
 *
 *  Built as a LIST rather than as conditional elements joined by a
 *  separator: an element that renders null is still an element, so
 *  Children.toArray would keep it and put a dot around nothing. Deciding
 *  here what exists means the separator only ever sits between two things
 *  the reader can see. */
function settingsParts(t: Team) {
  const parts: ReactNode[] = [];
  if (t.exclude_repos.length > 0) {
    parts.push(
      <>
        Excluded repos:{" "}
        <code className="font-mono text-[12px] text-muted">{t.exclude_repos.join(", ")}</code>
      </>,
    );
  }
  if (t.auto_review_authors > 0) {
    parts.push(
      <>
        Auto-review authors: <span className="text-muted">{t.auto_review_authors}</span>
      </>,
    );
  }
  if (t.ignore_authors > 0) {
    parts.push(
      <>
        Ignored authors: <span className="text-muted">{t.ignore_authors}</span>
      </>,
    );
  }
  return parts;
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

        {byName(data.teams).map((t) => (
          <Card
            key={t.slug}
            title={t.name}
            // The slug leads the note: it is the webhook path, the team= log
            // field and the settings route, so it has to stay on the page
            // once the heading shows the display name instead.
            note={
              <>
                {t.slug}
                <Sep />
                {t.enabled ? scope(t.repos) : "not started"}
                <Sep />
                {t.prs} PRs
                <Sep />
                last run {ago(t.last_run, now)}
              </>
            }
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
            {settingsParts(t).length > 0 && (
              <Note>
                {settingsParts(t).map((part, i) => (
                  <Fragment key={i}>
                    {i > 0 && <Sep />}
                    {part}
                  </Fragment>
                ))}
              </Note>
            )}
            {!t.enabled && (
              <Note>
                Look in the pod log for the{" "}
                <code className="font-mono text-[12px]">team_disabled</code> line to see why. It is
                not shown here because the reason can name an environment variable.
              </Note>
            )}
          </Card>
        ))}
        {data.teams.length === 0 && <Empty>No teams configured.</Empty>}
      </div>
    </div>
  );
}
