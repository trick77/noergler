import type { TFunction } from "i18next";
import type { ReactNode } from "react";
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

/** accountTone: a working account is ok; one Bitbucket or Jira did not
 *  answer may recover on its own; anything else waits for the team to push
 *  a working token. */
function accountTone(status: AccountStatus | null): Tone {
  if (status === "ok") return "ok";
  if (status === "unreachable") return "skip";
  return "fail";
}

/** integrationPills is the team's Bitbucket bot, Jira token and riptide,
 *  always all three: not set up (shared, off) is a state too, in the quiet
 *  tone, so a missing one never has to be inferred. Not the bot's name:
 *  what matters is whether the team runs on its own account and whether
 *  that works (user, 2026-10-26). The status is the server's fixed word,
 *  never a reason: this page is unauthenticated. */
function integrationPills(team: Team, now: number, t: TFunction) {
  const bb = team.bitbucket_account;
  const jr = team.jira_account;
  const suffix = (status: AccountStatus) =>
    status === "ok" ? "" : `: ${t(`teams.accountStatus.${status}`)}`;

  let bitbucket: ReactNode;
  if (bb?.source === "team") {
    const status = bb.status ?? "ok";
    bitbucket = (
      <Pill
        tone={accountTone(status)}
        title={t(`teams.bitbucketHint.${status}`, {
          slug: team.slug,
          ago: ago(bb.since, now),
        })}
      >
        {t("teams.ownBot")}
        {suffix(status)}
      </Pill>
    );
  } else {
    bitbucket = (
      <Pill tone="quiet" title={t("teams.sharedBotHint")}>
        {t("teams.sharedBot")}
      </Pill>
    );
  }

  let jira: ReactNode;
  if (jr?.source === "team") {
    const status = jr.status ?? "ok";
    const hint =
      status === "ok" || status === "unreachable" ? status : "broken";
    jira = (
      <Pill
        tone={accountTone(status)}
        title={t(`teams.jiraHint.${hint}`, {
          slug: team.slug,
          ago: ago(jr.since, now),
        })}
      >
        {t("teams.ownJira")}
        {suffix(status)}
      </Pill>
    );
  } else {
    jira = (
      <Pill tone="quiet" title={t("teams.sharedJiraHint")}>
        {t("teams.sharedJira")}
      </Pill>
    );
  }

  const rs = team.riptide ?? "off";
  const riptideTone: Record<typeof rs, Tone> = {
    ok: "ok",
    unreachable: "skip",
    rejected: "fail",
    off: "quiet",
  };
  const riptide = (
    <Pill tone={riptideTone[rs]} title={t(`teams.riptideHint.${rs}`)}>
      {t(`teams.riptide.${rs}`)}
    </Pill>
  );

  return { bitbucket, jira, riptide };
}

function ConfigGroup({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <h4 className="mb-1 text-[11px] font-medium tracking-[.1em] text-faint uppercase">
        {title}
      </h4>
      <dl className="grid grid-cols-[max-content_minmax(0,1fr)] text-[13px]">
        {children}
      </dl>
    </div>
  );
}

/** ConfigRow wraps its value, pills included: a pill that keeps to one line
 *  runs past the card on a narrow column. */
function ConfigRow({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <>
      <dt className="border-b border-border-soft py-1.5 pr-4 text-muted">
        {label}
      </dt>
      <dd className="m-0 min-w-0 border-b border-border-soft py-1.5 text-ink-dim [overflow-wrap:anywhere] [&>span]:whitespace-normal">
        {children}
      </dd>
    </>
  );
}

/** Configuration is collapsed by default; its summary keeps the three
 *  integration pills, so a closed card still answers "own bot? riptide?".
 *  Every row shows, set or not: an empty filter says so rather than
 *  vanishing. The author lists are COUNTS, never names: a roster of who
 *  works where is a different question from "is this team configured". */
function Configuration({ team, now }: { team: Team; now: number }) {
  const { t } = useTranslation();
  const pills = integrationPills(team, now, t);
  const unset = (word: string) => <span className="text-faint">{word}</span>;
  const globs = (list: string[]) =>
    list.length > 0 ? (
      <code className="font-mono text-[12px]">{list.join(", ")}</code>
    ) : (
      unset(t("teams.noneValue"))
    );
  const count = (n: number, empty: string) =>
    n > 0 ? <span className="tabular-nums">{n}</span> : unset(empty);

  return (
    <details className="group mt-3.5 border-t border-border pt-2.5">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-2.5 gap-y-1.5 rounded-ui-sm py-0.5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent [&::-webkit-details-marker]:hidden">
        <span
          aria-hidden
          className="h-[7px] w-[7px] -rotate-45 border-r-[1.5px] border-b-[1.5px] border-muted transition-transform group-open:rotate-45 motion-reduce:transition-none"
        />
        <span className="text-[11px] font-medium tracking-[.12em] text-faint uppercase group-hover:text-ink-dim">
          {t("teams.configuration")}
        </span>
        <span className="flex flex-wrap gap-1.5 group-open:hidden">
          {pills.bitbucket}
          {pills.jira}
          {pills.riptide}
        </span>
      </summary>
      <div className="mt-2.5 grid gap-x-8 gap-y-4 md:grid-cols-2">
        <ConfigGroup title={t("teams.integrations")}>
          <ConfigRow label={t("teams.model")}>
            {team.model ? (
              <>
                <code className="font-mono text-[12px]">{team.model}</code>
                <span
                  className={
                    "ml-2 text-[11px] " +
                    (team.model_source === "team"
                      ? "text-accent"
                      : "text-faint")
                  }
                >
                  {t(`teams.modelSource.${team.model_source ?? "default"}`)}
                </span>
              </>
            ) : (
              unset(t("teams.noneValue"))
            )}
          </ConfigRow>
          <ConfigRow label={t("teams.bitbucketBot")}>
            {pills.bitbucket}
          </ConfigRow>
          <ConfigRow label={t("teams.jiraToken")}>{pills.jira}</ConfigRow>
          <ConfigRow label={t("teams.riptideLabel")}>{pills.riptide}</ConfigRow>
        </ConfigGroup>
        <ConfigGroup title={t("teams.filters")}>
          <ConfigRow label={t("teams.excludedRepos")}>
            {globs(team.exclude_repos)}
          </ConfigRow>
          <ConfigRow label={t("teams.excludedSourceBranches")}>
            {globs(team.exclude_source_branches)}
          </ConfigRow>
          <ConfigRow label={t("teams.excludedTargetBranches")}>
            {globs(team.exclude_target_branches)}
          </ConfigRow>
          {/* An empty auto-review list means every author is reviewed. */}
          <ConfigRow label={t("teams.autoReviewAuthors")}>
            {count(team.auto_review_authors, t("teams.allAuthors"))}
          </ConfigRow>
          <ConfigRow label={t("teams.ignoredAuthors")}>
            {count(team.ignore_authors, t("teams.noneValue"))}
          </ConfigRow>
        </ConfigGroup>
      </div>
    </details>
  );
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
            // never known, so a summary could only repeat or guess. The
            // model lives in the configuration list.
            note={
              <>
                {t("teams.prs", { count: team.prs })}
                <Sep />
                {t("teams.lastReviewed", { ago: ago(team.last_reviewed, now) })}
                <Sep />
                {t("teams.lastRun", { ago: ago(team.last_run, now) })}
              </>
            }
            right={
              <Pill
                tone={teamTone(team.state)}
                title={teamStateTitle(team.state)}
              >
                {teamStateLabel(team.state)}
              </Pill>
            }
          >
            {team.claims.length === 0 ? (
              <Empty>{t("teams.noClaims")}</Empty>
            ) : (
              <table className={table}>
                <thead>
                  <tr>
                    <th className={th}>{t("teams.claim")}</th>
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
            {team.enabled ? (
              <Configuration team={team} now={now} />
            ) : (
              // A disabled team has no snapshot, so nothing to configure is
              // known; the log line is the only pointer.
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
