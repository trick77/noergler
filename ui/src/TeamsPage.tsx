import type { TFunction } from "i18next";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent,
  type ReactNode,
} from "react";
import { Trans, useTranslation } from "react-i18next";
import type { AccountStatus, ForwardStatus, Team } from "./api";
import {
  ago,
  byName,
  teamStateLabel,
  teamStateTitle,
  teamTone,
  type Tone,
} from "./format";
import { onNav, teamPath, teamSlugFromPath } from "./routing";
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

/** forwarderTone is the tone of a forwarder's status word, riptide and
 *  Almanach alike: a refused token is a fault, a failed call an outage. */
const forwarderTone: Record<ForwardStatus, Tone> = {
  ok: "ok",
  failing: "skip",
  rejected: "fail",
  off: "quiet",
};

/** accountTone: a working account is ok; one Bitbucket or Jira did not
 *  answer may recover on its own; anything else waits for the team to push
 *  a working token. */
function accountTone(status: AccountStatus | null): Tone {
  if (status === "ok") return "ok";
  if (status === "unreachable") return "skip";
  return "fail";
}

/** integrationPills is the team's Bitbucket bot, Jira token, riptide and
 *  Almanach, always all four: not set up (shared, off) is a state too, in the quiet
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
  const riptide = (
    <Pill tone={forwarderTone[rs]} title={t(`teams.riptideHint.${rs}`)}>
      {t(`teams.riptide.${rs}`)}
    </Pill>
  );

  // Almanach is a forwarder like riptide: same words, same tones.
  const as = team.almanach ?? "off";
  const almanach = (
    <Pill tone={forwarderTone[as]} title={t(`teams.almanachHint.${as}`)}>
      {t(`teams.almanach.${as}`)}
    </Pill>
  );

  return { bitbucket, jira, riptide, almanach };
}

// The search. Client-side: /teams is the whole roster in one response, so
// the filter needs no server round trip (unlike the Runs feed, a window).
// A hit is a case-insensitive substring of the team's name, its slug, a
// claim (project or PROJECT/repo) or an excluded-repo glob: the last so
// "why does nobody review contenthub-code?" finds the team that carved it
// out, with its configuration opened on the hit.

function claimText(c: { project: string; repo?: string }): string {
  return c.repo ? `${c.project}/${c.repo}` : c.project;
}

function has(text: string, q: string): boolean {
  return text.toLowerCase().includes(q);
}

/** excludeHit is the excluded-repo glob the query hits, "" if none. */
function excludeHit(team: Team, q: string): string {
  if (q === "") return "";
  return team.exclude_repos.find((g) => has(g, q)) ?? "";
}

/** matchedClaim is the claim the query hits, "" if none. */
function matchedClaim(team: Team, q: string): string {
  if (q === "") return "";
  const c = team.claims.find((c) => has(claimText(c), q));
  return c ? claimText(c) : "";
}

function matches(team: Team, q: string): boolean {
  if (q === "") return true;
  return (
    has(team.name, q) ||
    has(team.slug, q) ||
    matchedClaim(team, q) !== "" ||
    excludeHit(team, q) !== ""
  );
}

/** Mark is `text` with the query's first hit highlighted. */
function Mark({ text, q }: { text: string; q: string }) {
  const i = q === "" ? -1 : text.toLowerCase().indexOf(q);
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark className="rounded-[3px] bg-ochre-wash px-px text-ochre">
        {text.slice(i, i + q.length)}
      </mark>
      {text.slice(i + q.length)}
    </>
  );
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

/** Configuration is collapsed by default; its summary keeps the four
 *  integration pills, so a closed card still answers "own bot? riptide?
 *  Almanach?". Open for the team a link names and for a search hit in the
 *  excluded repos, which is otherwise behind the fold.
 *  Every row shows, set or not: an empty filter says so rather than
 *  vanishing. The author lists are COUNTS, never names: a roster of who
 *  works where is a different question from "is this team configured". */
function Configuration({
  team,
  now,
  q,
  open,
}: {
  team: Team;
  now: number;
  q: string;
  open: boolean;
}) {
  const { t } = useTranslation();
  const pills = integrationPills(team, now, t);
  const unset = (word: string) => <span className="text-faint">{word}</span>;
  // One string, marked as one: the hit may span a comma.
  const globs = (list: string[], mark = false) =>
    list.length > 0 ? (
      <code className="font-mono text-[12px]">
        {mark ? <Mark text={list.join(", ")} q={q} /> : list.join(", ")}
      </code>
    ) : (
      unset(t("teams.noneValue"))
    );
  const count = (n: number, empty: string) =>
    n > 0 ? <span className="tabular-nums">{n}</span> : unset(empty);

  return (
    <details open={open} className="group mt-3.5 border-t border-border pt-2.5">
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
          {pills.almanach}
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
          <ConfigRow label={t("teams.almanachLabel")}>
            {pills.almanach}
          </ConfigRow>
        </ConfigGroup>
        <ConfigGroup title={t("teams.filters")}>
          <ConfigRow label={t("teams.excludedRepos")}>
            {globs(team.exclude_repos, true)}
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

/** CopyLink is the # after a card's title: invisible until the header is
 *  hovered or it has focus, it copies the team's own link and says so for a
 *  moment. The slug surfaces here, in the link, and nowhere beside the
 *  name (user, 2026-10-26). */
function CopyLink({ team }: { team: Team }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const path = teamPath(team.slug);
  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1200);
    return () => clearTimeout(id);
  }, [copied]);
  // "copied" only once it is: the clipboard API is absent on a plain-http
  // host and may refuse; execCommand is the fallback that still works there.
  const copy = () => {
    const text = window.location.origin + path;
    const fallback = () => {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      let ok: boolean;
      try {
        ok = document.execCommand("copy");
      } catch {
        ok = false;
      }
      ta.remove();
      setCopied(ok);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard
        .writeText(text)
        .then(() => setCopied(true))
        .catch(fallback);
    } else {
      fallback();
    }
  };
  return (
    <button
      type="button"
      onClick={copy}
      aria-label={t("teams.copyLink", { name: team.name, path })}
      title={t("teams.copyLink", { name: team.name, path })}
      className={
        "-ml-1 rounded-md px-1.5 py-0.5 font-mono text-[15px] leading-none transition-opacity hover:bg-bg hover:text-accent-strong focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-accent " +
        (copied
          ? "font-sans text-[12px] text-[#82c491] opacity-100"
          : "text-faint opacity-0 group-hover/card:opacity-100 [@media(hover:none)]:opacity-100")
      }
    >
      {copied ? t("teams.copied") : "#"}
    </button>
  );
}

/** The rail's state groups, in the order the page reads them. A state this
 *  build has no word for still gets a group, after these, so no team hides. */
const STATES = ["ready", "no_repos", "disabled"];

function dotClass(tone: Tone): string {
  if (tone === "ok") return "bg-online";
  if (tone === "skip") return "bg-ochre";
  return "bg-danger";
}

/** Rail is the team index beside the cards: search on top, then every team
 *  as a row under its state, PR count on the right. A row that misses the
 *  query is dimmed, not dropped: the roster stays whole, and the group
 *  count says how many hit. A hit on a claim or an excluded repo is named
 *  under the row, since the name alone does not say why it matched. On a
 *  narrow screen the groups give way to a strip of the names. */
function Rail({
  teams,
  q,
  query,
  setQ,
  current,
  pick,
}: {
  teams: Team[];
  /** The input's text as typed. */
  q: string;
  /** The same, trimmed and lower-cased: what the cards match on too. */
  query: string;
  setQ: (q: string) => void;
  current: string | null;
  pick: (slug: string) => (e: MouseEvent) => void;
}) {
  const { t } = useTranslation();
  const input = useRef<HTMLInputElement>(null);
  // "/" focuses the search from anywhere on the page, Escape clears it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = input.current;
      if (!el) return;
      if (e.key === "/" && document.activeElement !== el) {
        e.preventDefault();
        el.focus();
        el.select();
      } else if (e.key === "Escape" && document.activeElement === el) {
        setQ("");
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [setQ]);

  const states = [
    ...STATES,
    ...teams.map((x) => x.state).filter((s) => !STATES.includes(s)),
  ].filter((s, i, all) => all.indexOf(s) === i);
  // Once per team: whether it hits, and why when the name does not say.
  const hits = new Map(
    teams.map((x) => {
      const hit = matches(x, query);
      const byName = has(x.name, query) || has(x.slug, query);
      const claim = byName ? "" : matchedClaim(x, query);
      const glob = byName || claim ? "" : excludeHit(x, query);
      return [x.slug, { hit, claim, glob }];
    }),
  );
  const shown = teams.filter((x) => hits.get(x.slug)?.hit).length;

  return (
    <nav
      aria-label={t("teams.railLabel")}
      className="lg:sticky lg:top-0 lg:self-start"
    >
      <label className="relative mb-3 flex items-center">
        <svg
          aria-hidden
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          className="pointer-events-none absolute left-2.5 h-3.5 w-3.5 text-faint"
        >
          <circle cx="7" cy="7" r="4.5" />
          <path d="m10.5 10.5 3.5 3.5" />
        </svg>
        <input
          ref={input}
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={t("teams.searchPlaceholder")}
          aria-label={t("teams.searchLabel")}
          autoComplete="off"
          spellCheck={false}
          className="w-full rounded-ui-sm border border-border bg-panel py-1.5 pr-8 pl-8 text-[14px] text-ink outline-none placeholder:text-faint focus:border-accent-strong [&::-webkit-search-cancel-button]:appearance-none"
        />
        {q === "" ? (
          <kbd
            aria-hidden
            className="absolute right-2 rounded-[5px] border border-border bg-bg px-1.5 font-mono text-[11px] text-faint"
          >
            /
          </kbd>
        ) : (
          <button
            type="button"
            onClick={() => setQ("")}
            aria-label={t("teams.clearSearch")}
            className="absolute right-2 rounded-[5px] px-1.5 text-[12px] text-muted hover:bg-active hover:text-ink"
          >
            ×
          </button>
        )}
      </label>

      <div className="flex gap-1 overflow-x-auto pb-1 [scrollbar-width:none] lg:block lg:overflow-visible lg:pb-0">
        {states.map((s) => {
          const group = teams.filter((x) => x.state === s);
          if (group.length === 0) return null;
          const n = group.filter((x) => hits.get(x.slug)?.hit).length;
          return (
            <div key={s} className="contents lg:block">
              <div className="mt-3 mb-1 ml-2.5 hidden items-center gap-1.5 text-[11px] font-medium tracking-[.12em] text-faint uppercase lg:flex">
                <span
                  aria-hidden
                  className={
                    "h-1.5 w-1.5 rounded-full " + dotClass(teamTone(s))
                  }
                />
                {teamStateLabel(s)}
                <span className="font-mono text-[12px] tracking-normal tabular-nums">
                  {query === "" ? group.length : `${n}/${group.length}`}
                </span>
              </div>
              {group.map((x) => {
                const { hit, claim, glob } = hits.get(x.slug)!;
                return (
                  <a
                    key={x.slug}
                    href={teamPath(x.slug)}
                    onClick={pick(x.slug)}
                    aria-current={x.slug === current ? "true" : undefined}
                    className={
                      "flex shrink-0 items-baseline gap-2 rounded-ui-sm border border-border px-2.5 py-1 text-[13.5px] whitespace-nowrap hover:bg-active hover:text-ink aria-[current]:bg-active aria-[current]:text-ink lg:block lg:border-0 lg:py-1.5 lg:whitespace-normal lg:aria-[current]:shadow-[inset_2px_0_0_var(--color-accent-strong)] " +
                      (hit ? "text-muted" : "text-muted opacity-40")
                    }
                  >
                    <span className="flex items-baseline gap-2">
                      <span className="overflow-hidden text-ellipsis">
                        <Mark text={x.name} q={query} />
                      </span>
                      {x.prs > 0 && (
                        <span className="ml-auto hidden font-mono text-[12px] text-faint tabular-nums lg:inline">
                          {x.prs}
                        </span>
                      )}
                    </span>
                    {/* The label stays outside the mark: only the glob
                        matched, never the word "excluded". */}
                    {(claim || glob) && (
                      <span className="mt-0.5 hidden overflow-hidden font-mono text-[11.5px] text-ellipsis whitespace-nowrap text-faint lg:block">
                        ↳{" "}
                        {claim ? (
                          <Mark text={claim} q={query} />
                        ) : (
                          <Trans
                            i18nKey="teams.excludedHit"
                            components={{
                              glob: <Mark text={glob} q={query} />,
                            }}
                          />
                        )}
                      </span>
                    )}
                  </a>
                );
              })}
            </div>
          );
        })}
      </div>
      <p className="mt-3.5 ml-2.5 hidden text-[12.5px] text-faint lg:block">
        {query === ""
          ? t("teams.count", { count: teams.length })
          : t("teams.matchCount", { shown, total: teams.length })}
      </p>
    </nav>
  );
}

/** cardId is a team card's DOM id. Prefixed: a bare slug is page-global
 *  and `root`, a valid slug, is the app's mount. */
const CARD_ID = "team-";
function cardId(slug: string): string {
  return CARD_ID + slug;
}

/** readQuery is the search the URL carries (?q=), so a filtered view can be
 *  shared; writeQuery keeps the URL in step without a history entry per
 *  keystroke. */
function readQuery(): string {
  return new URLSearchParams(window.location.search).get("q") ?? "";
}
function writeQuery(path: string, q: string) {
  const url = q === "" ? path : `${path}?q=${encodeURIComponent(q)}`;
  if (window.location.pathname + window.location.search !== url) {
    window.history.replaceState({}, "", url);
  }
}

export function TeamsPage() {
  const { t } = useTranslation();
  const { data, failed } = usePoll<{ teams: Team[] }>("teams", 30000);
  const now = useNow(30000);
  const [q, setQuery] = useState(readQuery);
  // The team the path names: scrolled to, ringed, its configuration open.
  const [focus, setFocus] = useState(() =>
    teamSlugFromPath(window.location.pathname),
  );
  // The card in view, for the rail's marker; the focus until the reader
  // scrolls away from it.
  const [current, setCurrent] = useState(focus);
  // Bumped by a rail click on the team already in the path, which changes
  // nothing else and still has to scroll.
  const [again, setAgain] = useState(0);
  const scroller = useRef<HTMLDivElement>(null);
  const query = q.trim().toLowerCase();

  // Back/forward between two teams' links, and navigateTo's own event: the
  // URL is the source, the state follows it.
  useEffect(() => {
    const onPop = () => {
      const slug = teamSlugFromPath(window.location.pathname);
      setFocus(slug);
      setCurrent(slug);
      setQuery(readQuery());
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // Stable: the rail's key listener subscribes on it.
  const setQ = useCallback((next: string) => {
    setQuery(next);
    // A search is a view of the whole page, so the path drops the team.
    setFocus(null);
    writeQuery("/teams", next);
  }, []);

  // Scroll to the focused team once its card exists. Keyed on the focus,
  // not the data: a poll that re-renders must not yank the reader back.
  const loaded = data !== null;
  useEffect(() => {
    if (!loaded || focus === null) return;
    // A slug is [a-z0-9-], so the selector needs no escaping.
    const el = scroller.current?.querySelector<HTMLElement>(
      `#${cardId(focus)}`,
    );
    // Optional: jsdom has no scrollIntoView.
    el?.scrollIntoView?.({ block: "start" });
  }, [loaded, focus, again]);

  // The rail click is an in-app link (onNav: a modified click stays the
  // browser's); the team already open gets its scroll by hand, since the
  // path does not change.
  const pick = (slug: string) => (e: MouseEvent) => {
    const path = teamPath(slug);
    // Before onNav: afterwards the path is this one either way.
    const same = window.location.pathname === path;
    onNav(path)(e);
    if (e.isDefaultPrevented() && same) {
      setQ("");
      setFocus(slug);
      setAgain((n) => n + 1);
    }
  };

  // The card at the top of the viewport marks its rail row, so the rail
  // doubles as a scroll position. Not while searching: the rows then say
  // what matched, which is the more useful thing.
  const onScroll = () => {
    const root = scroller.current;
    if (!root || query !== "") return;
    const top = root.getBoundingClientRect().top + 80;
    const cards = Array.from(
      root.querySelectorAll<HTMLElement>(`section[id^="${CARD_ID}"]`),
    );
    const at =
      cards.reverse().find((c) => c.getBoundingClientRect().top <= top) ??
      cards[cards.length - 1];
    const slug = at ? at.id.slice(CARD_ID.length) : null;
    if (slug !== current) setCurrent(slug);
  };

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

  const teams = byName(data.teams);
  const shown = teams.filter((team) => matches(team, query));
  const unknown = focus !== null && !teams.some((x) => x.slug === focus);

  return (
    <div ref={scroller} onScroll={onScroll} className={page}>
      <div className={column}>
        <h2 className={h2}>{t("nav.teams")}</h2>

        {teams.length === 0 ? (
          <Empty>{t("teams.none")}</Empty>
        ) : (
          <div className="mt-5 flex flex-col gap-3 lg:grid lg:grid-cols-[240px_minmax(0,1fr)] lg:items-start lg:gap-7">
            <Rail
              teams={teams}
              q={q}
              query={query}
              setQ={setQ}
              current={current}
              pick={pick}
            />
            <div className="min-w-0">
              {unknown && (
                <Note>
                  <Trans
                    i18nKey="teams.unknownTeam"
                    values={{ slug: focus }}
                    components={{
                      code: <code className="font-mono text-[12px]" />,
                    }}
                  />
                </Note>
              )}
              {shown.length === 0 && (
                <p className="py-7 text-center text-[13.5px] text-faint">
                  {/* The query is a React child, never interpolated into
                      the string: typed angle brackets stay text. */}
                  {t("teams.noMatch")}{" "}
                  <b className="font-medium text-ink-dim">{q.trim()}</b>.{" "}
                  <button
                    type="button"
                    onClick={() => setQ("")}
                    className="text-muted underline hover:text-ink"
                  >
                    {t("teams.showAll")}
                  </button>
                </p>
              )}
              {shown.map((team) => (
                <TeamCard
                  key={team.slug}
                  team={team}
                  now={now}
                  q={query}
                  focus={team.slug === focus}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TeamCard({
  team,
  now,
  q,
  focus,
}: {
  team: Team;
  now: number;
  q: string;
  focus: boolean;
}) {
  const { t } = useTranslation();
  return (
    <Card
      id={cardId(team.slug)}
      title={team.name}
      hint={team.slug}
      titleMark={<Mark text={team.name} q={q} />}
      afterTitle={<CopyLink team={team} />}
      // The id is the anchor and the scroll spy's key.
      className={(focus ? "ring-settle " : "") + "scroll-mt-4"}
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
        <Pill tone={teamTone(team.state)} title={teamStateTitle(team.state)}>
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
            {team.claims.map((c) => {
              const text = claimText(c);
              const hit = q !== "" && has(text, q);
              return (
                <tr
                  key={text}
                  className={hit ? "[&>td]:bg-ochre-wash/40" : undefined}
                >
                  <td className={tdTag}>
                    <Mark text={text} q={q} />
                  </td>
                  <td className={tdNum}>
                    {c.repo ? t("teams.scopeRepo") : t("teams.scopeProject")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {team.enabled ? (
        <Configuration
          team={team}
          now={now}
          q={q}
          open={focus || excludeHit(team, q) !== ""}
        />
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
  );
}
