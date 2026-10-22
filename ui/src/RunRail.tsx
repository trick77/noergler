// The rail: a pull request's runs, newest first, kept beside whichever view
// is open so a reader always sees where they are and reaches any run in one
// click. Every entry is a link to the same paths the tables use
// (/prs/{token}, /prs/{token}/runs/{n}); the replies and answers, which have
// no run and no page, are a view of the PR (#calls).
//
// On a narrow screen the rail is a strip of chips above the pane: the
// name, the outcome and, when the history is cut, that it is cut.
import type { MouseEvent } from "react";
import { useTranslation } from "react-i18next";
import type { PRPage as PRPageData, PRRun } from "./api";
import {
  duration,
  money,
  outcomeTitle,
  outcomeTone,
  outcomeWord,
  when,
} from "./format";
import type { PRView } from "./PRPage";
import { onNav, prPath } from "./routing";
import { Pill, Sep } from "./ui";

// A bordered card with a chevron, so an entry reads as a button at rest;
// the selected one carries the accent border and no chevron. The chevron is
// an element, aria-hidden: as ::after content it would be read as part of
// every link's name.
const entry =
  "group relative block shrink-0 rounded-ui-sm border border-border bg-panel px-3 py-2 transition-colors hover:bg-active aria-[current=page]:border-accent-strong " +
  "lg:mb-1.5 lg:pr-8";
const row = "flex items-center gap-2";
const name = "font-medium whitespace-nowrap text-ink";
const sub = "mt-0.5 hidden text-[12.5px] text-muted lg:block";
const label =
  "mt-4 mb-1.5 ml-3 hidden items-center gap-1.5 text-[11px] font-medium tracking-[.12em] text-faint uppercase lg:flex";

export function RunRail({
  d,
  token,
  n,
  view,
  pickView,
}: {
  d: PRPageData;
  token: string;
  /** The open run, null on the PR's own views. */
  n: number | null;
  view: PRView;
  pickView: (v: PRView) => void;
}) {
  const { t } = useTranslation();
  const current = (on: boolean) => (on ? ("page" as const) : undefined);
  // A PR view from anywhere: back to the PR's path if a run is open, then
  // the view's hash. onNav decides whether the click is the SPA's or the
  // browser's (new tab); only a click it took changes the view.
  const toPR = (v: PRView) => (e: MouseEvent) => {
    onNav(prPath(token))(e);
    if (e.isDefaultPrevented()) pickView(v);
  };
  // Newest first: the run a reader came for is usually the latest.
  const runs = [...d.runs].reverse();
  // Nothing to list and nothing to report: no entry. A failed read still
  // gets one, so the pane can say so.
  const calls = d.calls_failed || d.calls.length > 0;

  return (
    <nav
      aria-label={t("pr.railLabel")}
      className="flex gap-1.5 overflow-x-auto [scrollbar-width:none] lg:sticky lg:top-0 lg:block lg:overflow-visible"
    >
      <a
        href={prPath(token)}
        onClick={toPR("findings")}
        aria-current={current(n === null && view !== "calls")}
        className={entry}
      >
        <Chevron />
        <div className={row}>
          <span className={name}>{t("pr.overview")}</span>
        </div>
        <div className={sub}>{t("pr.overviewSub")}</div>
      </a>

      <div className={label}>
        {t("pr.railRuns")}
        <span className="rounded-full bg-active px-1.5 font-mono text-[12px] tracking-normal text-ink-dim">
          {d.runs_total}
        </span>
      </div>
      {runs.map((r) => (
        <RunEntry key={r.n} r={r} token={token} current={n === r.n} />
      ))}
      {/* Said on the strip too: a cut list with no sign of it reads as
          the whole history. */}
      {d.more_runs && (
        <div className="shrink-0 self-center text-[12px] text-faint italic lg:mx-3 lg:mt-2">
          {t("pr.moreRuns")}
        </div>
      )}

      {calls && (
        <>
          <div className={label}>{t("pr.alsoOnPR")}</div>
          <a
            href={`${prPath(token)}#calls`}
            onClick={toPR("calls")}
            aria-current={current(n === null && view === "calls")}
            className={entry}
          >
            <Chevron />
            <div className={row}>
              <span className={name}>{t("pr.callsTitle")}</span>
              {/* Cut at its cap: "200+", never a count that reads as all
                  of them; none at all when the read failed, where a 0
                  would read as none. */}
              {d.calls.length > 0 && (
                <span className="ml-auto font-mono text-[12px] text-faint">
                  {d.more_calls ? `${d.calls.length}+` : d.calls.length}
                </span>
              )}
            </div>
            <div className={sub}>{t("pr.callsSub")}</div>
          </a>
        </>
      )}
    </nav>
  );
}

function Chevron() {
  return (
    <span
      aria-hidden
      className="absolute top-1.5 right-3 hidden text-[16px] text-faint lg:inline lg:group-aria-[current=page]:hidden"
    >
      ›
    </span>
  );
}

function RunEntry({
  r,
  token,
  current,
}: {
  r: PRRun;
  token: string;
  current: boolean;
}) {
  const { t } = useTranslation();
  const href = prPath(token, r.n);
  return (
    <a
      href={href}
      onClick={onNav(href)}
      aria-current={current ? "page" : undefined}
      className={entry}
    >
      <Chevron />
      <div className={row}>
        {/* A run that did not review is dimmed, the sequence stays whole. */}
        <span className={r.run ? name : "whitespace-nowrap text-muted"}>
          {t("run.crumb", { n: r.n })}
        </span>
        <Pill
          tone={outcomeTone(r.outcome)}
          title={outcomeTitle(r.outcome, r.reason, r.reason_label)}
        >
          {outcomeWord(r.outcome, r.reason)}
        </Pill>
        <span className="ml-auto hidden font-mono text-[12px] whitespace-nowrap text-faint lg:inline">
          {when(r.created_at)}
        </span>
      </div>
      <div className={sub}>
        {r.kind === "mention" ? t("run.mention") : t("run.auto")}
        <Sep />
        {r.run ? (
          <>
            {r.run.incremental ? t("run.incremental") : t("run.full")}
            <Sep />
            {t("pr.scopeFiles", { count: r.run.files_changed })}{" "}
            <span className="font-mono">
              +{r.run.lines_added}/−{r.run.lines_removed}
            </span>
          </>
        ) : (
          duration(r.elapsed_ms)
        )}
      </div>
      {r.run && (
        <div className="mt-1 hidden flex-wrap gap-x-2.5 font-mono text-[12px] text-faint tabular-nums lg:flex">
          <span>{t("pr.railNew", { count: r.run.findings_posted })}</span>
          <span>{t("pr.railFixed", { count: r.fixed })}</span>
          <span>{duration(r.elapsed_ms)}</span>
          {r.cost_usd !== null && <span>{money(r.cost_usd)}</span>}
        </div>
      )}
    </a>
  );
}
