import type { TFunction } from "i18next";
import { Fragment, type ReactNode } from "react";
import { Trans, useTranslation } from "react-i18next";
import type { AccountStatus, Team } from "./api";
import {
  ago,
  byName,
  teamStateLabel,
  teamStateTitle,
  teamTone,
  type Tone,
} from "./format";
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
function settingsParts(team: Team, t: TFunction) {
  const parts: ReactNode[] = [];
  if (team.exclude_repos.length > 0) {
    parts.push(
      <>
        {t("teams.excludedRepos")}{" "}
        <code className="font-mono text-[12px] text-muted">
          {team.exclude_repos.join(", ")}
        </code>
      </>,
    );
  }
  if (team.exclude_source_branches.length > 0) {
    parts.push(
      <>
        {t("teams.excludedSourceBranches")}{" "}
        <code className="font-mono text-[12px] text-muted">
          {team.exclude_source_branches.join(", ")}
        </code>
      </>,
    );
  }
  if (team.exclude_target_branches.length > 0) {
    parts.push(
      <>
        {t("teams.excludedTargetBranches")}{" "}
        <code className="font-mono text-[12px] text-muted">
          {team.exclude_target_branches.join(", ")}
        </code>
      </>,
    );
  }
  if (team.auto_review_authors > 0) {
    parts.push(
      <>
        {t("teams.autoReviewAuthors")}{" "}
        <span className="text-muted">{team.auto_review_authors}</span>
      </>,
    );
  }
  if (team.ignore_authors > 0) {
    parts.push(
      <>
        {t("teams.ignoredAuthors")}{" "}
        <span className="text-muted">{team.ignore_authors}</span>
      </>,
    );
  }
  return parts;
}

/** accountTone: a working account is ok; one Bitbucket or Jira did not
 *  answer may recover on its own; anything else waits for the team to push
 *  a working token. */
function accountTone(status: AccountStatus | null): Tone {
  if (status === "ok") return "ok";
  if (status === "unreachable") return "skip";
  return "fail";
}

/** AccountPills tags a team on its OWN Bitbucket bot or Jira token, beside
 *  the state pill; a team on the shared accounts gets nothing. Not the bot's
 *  name: what matters here is that the team runs on its own account and
 *  whether that works (user, 2026-10-26). The status is the server's fixed
 *  word, never a reason: this page is unauthenticated. */
function AccountPills({ team, now }: { team: Team; now: number }) {
  const { t } = useTranslation();
  const bb = team.bitbucket_account;
  const jr = team.jira_account;
  const pills: ReactNode[] = [];
  if (bb?.source === "team") {
    const status = bb.status ?? "ok";
    pills.push(
      <Pill
        key="bitbucket"
        tone={accountTone(status)}
        title={t(`teams.bitbucketHint.${status}`, {
          slug: team.slug,
          ago: ago(bb.since, now),
        })}
      >
        {t("teams.teamBitbucket")}
        {status !== "ok" && `: ${t(`teams.accountStatus.${status}`)}`}
      </Pill>,
    );
  }
  if (jr?.source === "team") {
    const status = jr.status ?? "ok";
    pills.push(
      <Pill
        key="jira"
        tone={accountTone(status)}
        title={t(
          `teams.jiraHint.${status === "ok" || status === "unreachable" ? status : "broken"}`,
          {
            slug: team.slug,
            ago: ago(jr.since, now),
          },
        )}
      >
        {t("teams.teamJira")}
        {status !== "ok" && `: ${t(`teams.accountStatus.${status}`)}`}
      </Pill>,
    );
  }
  return <>{pills}</>;
}

export function TeamsPage() {
  const { t } = useTranslation();
  const { data, failed } = usePoll<{ teams: Team[] }>("teams", 30000);
  const now = useNow(30000);

  if (!data) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>{t("nav.teams")}</h2>
          {failed ? (
            <Failed what={t("teams.failedWhat")} />
          ) : (
            <div className="skeleton h-24 rounded-ui" />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>{t("nav.teams")}</h2>

        {byName(data.teams).map((team) => (
          <Card
            key={team.slug}
            title={team.name}
            hint={team.slug}
            // The display name only, never the slug beside it: both read as
            // the same team twice (user, 2026-10-26). The slug stays on
            // hover, as on TeamPill: it is the team= log field the disabled
            // note points at, and the webhook path. No scope here: the
            // claims table below lists it, and a project's repo count is
            // never known, so a summary could only repeat or guess.
            note={
              <>
                {team.model && (
                  <>
                    <code className="font-mono text-[12px]">{team.model}</code>
                    <Sep />
                  </>
                )}
                {t("teams.prs", { count: team.prs })}
                <Sep />
                {t("teams.lastReviewed", { ago: ago(team.last_reviewed, now) })}
                <Sep />
                {t("teams.lastRun", { ago: ago(team.last_run, now) })}
              </>
            }
            right={
              <span className="flex flex-wrap justify-end gap-1.5">
                <AccountPills team={team} now={now} />
                <Pill
                  tone={teamTone(team.state)}
                  title={teamStateTitle(team.state)}
                >
                  {teamStateLabel(team.state)}
                </Pill>
              </span>
            }
          >
            {team.claims.length === 0 ? (
              <Empty>{t("teams.noClaims")}</Empty>
            ) : (
              <table className={table}>
                <thead>
                  <tr>
                    <th className={th}>{t("teams.slug")}</th>
                    <th className={thNum}>{t("teams.scope")}</th>
                  </tr>
                </thead>
                <tbody>
                  {team.claims.map((c) => (
                    <tr key={`${c.project}/${c.repo ?? ""}`}>
                      <td className={tdTag}>
                        {c.repo ? `${c.project}/${c.repo}` : c.project}
                      </td>
                      <td className={tdNum}>
                        {c.repo
                          ? t("teams.scopeRepo")
                          : t("teams.scopeProject")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {settingsParts(team, t).length > 0 && (
              <Note>
                {settingsParts(team, t).map((part, i) => (
                  <Fragment key={i}>
                    {i > 0 && <Sep />}
                    {part}
                  </Fragment>
                ))}
              </Note>
            )}
            {!team.enabled && (
              <Note>
                <Trans
                  i18nKey="teams.disabledNote"
                  components={{
                    code: <code className="font-mono text-[12px]" />,
                  }}
                />
              </Note>
            )}
          </Card>
        ))}
        {data.teams.length === 0 && <Empty>{t("teams.none")}</Empty>}
      </div>
    </div>
  );
}
