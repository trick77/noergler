// The PR page: one shell for a pull request and its runs. The runs sit in a
// rail on the left on every view (RunRail), and the pane on the right is
// what is selected: the PR's overview (findings, files, cost), its replies
// and answers, or one run (RunPane, at /prs/{token}/runs/{n}). Reached only
// by its link token from the PR's summary comment.
//
// Figures add up where they can (cost is the cap's total, tokens and time are
// sums), and the whole PR's lines are the latest run's reading of the whole
// diff. Steps stay on each run's pane: across runs only the waits add up.
import { useEffect, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import type { CommentCall, PRFinding, PRPage as PRPageData } from "./api";
import {
  commentUrl,
  duration,
  known,
  money,
  prUrl,
  severityTone,
  severityWord,
  when,
} from "./format";
import { onNav, prPath } from "./routing";
import { RunPane } from "./RunPane";
import { RunRail } from "./RunRail";
import { FindingText } from "./RunPipeline";
import { usePoll } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Note,
  Pill,
  Sep,
  TeamPill,
  Tile,
  Tiles,
  column,
  h2,
  page,
  table,
  td,
  tdNum,
  tdTag,
  th,
  thNum,
} from "./ui";

/** PRView is what the pane shows for the PR itself: three tabs of the
 *  overview, or the replies and answers, which the rail selects. */
export type PRView = "findings" | "files" | "cost" | "calls";
type Tab = Exclude<PRView, "calls">;
const TABS: Tab[] = ["findings", "files", "cost"];
const VIEWS: PRView[] = [...TABS, "calls"];

/** The view lives in the hash (#files, #calls), so a link can open one and
 *  a reload keeps it. Findings first: what is still open is what a reader
 *  brings from the PR. */
function viewFromHash(): PRView {
  const h = window.location.hash.slice(1);
  return (VIEWS as string[]).includes(h) ? (h as PRView) : "findings";
}

export function PRPage({ token, n }: { token: string; n: number | null }) {
  const { t } = useTranslation();
  // A PR page changes with every run; the slow poll picks up the next one.
  // The shell is keyed by the token alone, so the rail keeps its data while
  // the pane switches between the PR and its runs.
  const { data, failed } = usePoll<PRPageData>(`prs/${token}`, 60000);
  const [view, setView] = useState<PRView>(viewFromHash);

  // The hash is the view. Read again whenever the path under the shell
  // changes (a rail click, the back button): a run's own hash is the run
  // pane's, and coming back to the PR starts from what the URL says.
  const [viewN, setViewN] = useState(n);
  if (viewN !== n) {
    setViewN(n);
    setView(viewFromHash());
  }
  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const pickView = (next: PRView) => {
    setView(next);
    window.history.replaceState(
      {},
      "",
      next === "findings" ? window.location.pathname : `#${next}`,
    );
  };

  const d = data;
  // The calls view with nothing to list (a stale #calls link) is the
  // overview: an empty card beside a rail with nothing selected says nothing.
  const hasCalls = d !== null && (d.calls_failed || d.calls.length > 0);
  const shown = view === "calls" && !hasCalls ? "findings" : view;
  const pr = d === null ? null : prUrl(d.tag, d.bitbucket_url);
  // A cut history does not reach the PR's first run: no start date then.
  const first = d === null || d.more_runs ? undefined : d.runs[0]?.created_at;
  const last = d?.runs[d.runs.length - 1]?.created_at;

  if (d === null && failed) {
    return (
      <div className={page}>
        <div className={column}>
          <h2 className={h2}>{t("pr.title")}</h2>
          <Failed what={t("pr.failedWhat")} />
        </div>
      </div>
    );
  }

  return (
    <div className={page}>
      <div className={column}>
        {/* The run pane fetches on its own below, so a run link's two reads
            run side by side: the PR's header and rail are a skeleton until
            theirs lands. */}
        {d === null ? (
          <>
            <h2 className={h2}>{t("pr.title")}</h2>
            <div className="skeleton mt-2 h-7 rounded-ui" />
          </>
        ) : (
          <>
            <nav className="text-[13px] text-faint" aria-label="breadcrumb">
              <a
                href="/runs"
                onClick={onNav("/runs")}
                className="text-muted hover:text-ink hover:underline"
              >
                {t("nav.runs")}
              </a>
              <Sep />
              <span className="font-mono">{d.tag}</span>
            </nav>
            <h2 className={h2 + " mt-2"}>{d.title || d.tag}</h2>
            <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1.5 text-[13.5px] text-muted">
              <Pill tone={d.state === "open" ? "ok" : "none"}>
                {known("prState", d.state) ? t(`prState.${d.state}`) : d.state}
              </Pill>
              <TeamPill name={d.team_name} slug={d.team} />
              <span>
                {t("pr.runs", { count: d.runs_total })}
                {d.calls.length > 0 && (
                  <>
                    <Sep />
                    {/* The list is cut at its cap: "200+", never a count that
                    reads as all of them. */}
                    {d.more_calls
                      ? t("pr.repliesMore", { count: d.calls.length })
                      : t("pr.replies", { count: d.calls.length })}
                  </>
                )}
              </span>
              {last && (
                <>
                  <Sep />
                  <span>
                    {!first || first === last
                      ? when(last)
                      : `${when(first)} → ${when(last)}`}
                  </span>
                </>
              )}
              <span className="flex gap-1 sm:ml-auto">
                {pr && (
                  <a
                    href={pr}
                    rel="noopener"
                    className="rounded-ui-sm border border-border px-2 py-0.5 text-[12.5px] whitespace-nowrap hover:bg-active hover:text-ink"
                  >
                    {t("run.openPR")}
                  </a>
                )}
              </span>
            </div>
          </>
        )}

        {/* The rail is a strip above the pane until there is room for a
            column beside it. */}
        <div className="mt-6 flex flex-col gap-4 lg:grid lg:grid-cols-[280px_minmax(0,1fr)] lg:items-start lg:gap-7">
          {d === null ? (
            <div className="skeleton h-40 rounded-ui" />
          ) : (
            <RunRail
              d={d}
              token={token}
              n={n}
              view={shown}
              pickView={pickView}
            />
          )}
          <div className="min-w-0">
            {n !== null ? (
              <RunPane key={n} token={token} n={n} />
            ) : d === null ? (
              <div className="skeleton h-24 rounded-ui" />
            ) : shown === "calls" ? (
              <CallsCard d={d} pr={pr} />
            ) : (
              <Overview
                d={d}
                pr={pr}
                token={token}
                tab={shown}
                pick={pickView}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Overview is the PR's own pane: its totals, then findings, files and
 *  cost in tabs. */
function Overview({
  d,
  pr,
  token,
  tab,
  pick,
}: {
  d: PRPageData;
  pr: string | null;
  token: string;
  tab: Tab;
  pick: (v: PRView) => void;
}) {
  const { t } = useTranslation();
  const fixed = d.findings.filter((f) => f.fixed).length;
  const cost = d.cost_usd !== null ? Number(d.cost_usd) : null;
  const cap = d.cap_usd !== null ? Number(d.cap_usd) : null;
  const tokens = d.prompt_tokens + d.completion_tokens;

  const counts: Record<Tab, number | null> = {
    findings: d.findings.length,
    files: d.files.length,
    cost: null,
  };
  const onTabKey = (e: KeyboardEvent, i: number) => {
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (step === 0) return;
    e.preventDefault();
    const n = TABS[(i + step + TABS.length) % TABS.length];
    pick(n);
    document.getElementById(`pr-tab-${n}`)?.focus();
  };

  return (
    <>
      <Tiles>
        <Tile label={t("pr.cost")} value={money(d.cost_usd)}>
          {cost !== null && cap !== null && cap > 0 && (
            <>
              <CapMeter cost={cost} cap={cap} />
              <div className="mt-1 text-[12px] text-faint">
                {t("pr.ofCap", { cap: money(d.cap_usd) })}
              </div>
            </>
          )}
        </Tile>
        <Tile
          label={t("pr.findings")}
          value={d.findings.length}
          note={t("pr.posted")}
        >
          {d.findings.length > 0 && (
            <div className="mt-0.5 text-[12px] text-faint">
              {t("pr.findingsNote", {
                fixed,
                open: d.findings.length - fixed,
              })}
            </div>
          )}
        </Tile>
        <Tile
          label={t("pr.scope")}
          value={
            d.scope === null ? (
              "—"
            ) : (
              <span className="font-mono text-[18px] sm:text-[21px]">
                +{d.scope.added} / −{d.scope.removed}
              </span>
            )
          }
        >
          {d.scope !== null && (
            <div className="mt-0.5 text-[12px] text-faint">
              {t("pr.scopeFiles", { count: d.scope.files })}
            </div>
          )}
        </Tile>
        <Tile
          label={t("pr.tokens")}
          value={tokens === 0 ? "—" : tokens.toLocaleString()}
        >
          <div className="mt-0.5 text-[12px] text-faint">
            {d.prompt_tokens > 0 &&
              t("pr.fromCache", {
                pct: Math.round((d.cached_tokens / d.prompt_tokens) * 100),
              }) + " · "}
            {/* The waits are read from the listed runs only: beside a
                whole-history time they would understate, so a cut
                history shows the time alone. */}
            {d.more_runs
              ? t("pr.timeOnly", { time: duration(d.elapsed_ms) })
              : t("pr.timeNote", {
                  time: duration(d.elapsed_ms),
                  wait: duration(d.wait_ms),
                })}
          </div>
        </Tile>
      </Tiles>

      <div
        role="tablist"
        aria-label={t("pr.tabsLabel")}
        className="sticky top-0 z-10 mt-6 mb-4 flex gap-0.5 overflow-x-auto border-b border-border bg-bg [scrollbar-width:none]"
      >
        {TABS.map((k, i) => (
          <button
            key={k}
            id={`pr-tab-${k}`}
            type="button"
            role="tab"
            aria-selected={tab === k}
            aria-controls={`pr-panel-${k}`}
            tabIndex={tab === k ? 0 : -1}
            onClick={() => pick(k)}
            onKeyDown={(e) => onTabKey(e, i)}
            className={
              "-mb-px inline-flex items-baseline gap-1.5 border-b-2 px-3 pt-2 pb-2.5 text-sm whitespace-nowrap transition-colors " +
              (tab === k
                ? "border-accent-strong text-ink"
                : "border-transparent text-muted hover:text-ink")
            }
          >
            {t(`pr.tab.${k}`)}
            {counts[k] !== null && (
              <span className="rounded-full bg-active px-1.5 font-mono text-[12px] text-ink-dim">
                {counts[k]}
              </span>
            )}
          </button>
        ))}
      </div>

      <div
        role="tabpanel"
        id={`pr-panel-${tab}`}
        aria-labelledby={`pr-tab-${tab}`}
      >
        {tab === "findings" && <FindingsTab d={d} pr={pr} token={token} />}
        {tab === "files" && <FilesTab d={d} token={token} />}
        {tab === "cost" && <CostTab d={d} />}
      </div>
    </>
  );
}

/** CapMeter is the PR's cost against its cap, red at or over it. */
function CapMeter({
  cost,
  cap,
  className = "mt-2 h-1.5 rounded-[4px]",
}: {
  cost: number;
  cap: number;
  className?: string;
}) {
  return (
    <div className={"overflow-hidden bg-track " + className}>
      <div
        className={"h-full " + (cost >= cap ? "bg-danger" : "bg-s1")}
        style={{ width: `${Math.min(100, (cost / cap) * 100)}%` }}
      />
    </div>
  );
}

/** RunLink is run n's pane, under its PR's: the same path the rail selects
 *  it by. */
function RunLink({ token, n }: { token: string; n: number }) {
  const { t } = useTranslation();
  const href = prPath(token, n);
  return (
    <a
      href={href}
      onClick={onNav(href)}
      className="text-muted hover:text-ink hover:underline"
    >
      {t("run.crumb", { n })}
    </a>
  );
}

function FindingsTab({
  d,
  pr,
  token,
}: {
  d: PRPageData;
  pr: string | null;
  token: string;
}) {
  const { t } = useTranslation();
  if (d.findings.length === 0) {
    return (
      <Card>
        <Empty>{t("pr.noFindings")}</Empty>
      </Card>
    );
  }
  // Open ones first: what is still to do leads.
  const rows = [...d.findings].sort(
    (a, b) => Number(a.fixed) - Number(b.fixed),
  );
  return (
    <Card title={t("pr.findingsTitle")}>
      <div className="overflow-x-auto overscroll-x-contain">
        <table className={table}>
          <thead>
            <tr>
              <th className={th + " w-px"}>{t("run.severity")}</th>
              <th className={th + " w-px"}>{t("run.where")}</th>
              <th className={th}>{t("run.finding")}</th>
              <th className={th + " w-px"}>{t("pr.postedIn")}</th>
              <th className={th + " w-px"}>{t("pr.now")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((f: PRFinding, i) => {
              const href = commentUrl(pr, f);
              return (
                <tr key={`${f.file}:${f.line}:${i}`}>
                  <td className={td}>
                    <Pill tone={severityTone(f.severity)}>
                      {severityWord(f.severity)}
                    </Pill>
                  </td>
                  <td className={tdTag}>
                    {href ? (
                      <a
                        href={href}
                        rel="noopener"
                        className="hover:text-ink hover:underline"
                      >
                        {f.file}:{f.line} ↗
                      </a>
                    ) : (
                      `${f.file}:${f.line}`
                    )}
                  </td>
                  {/* The headline only: the full text is on the PR. */}
                  <td className={td}>
                    <FindingText headline={f.headline} />
                  </td>
                  <td className={td + " whitespace-nowrap text-[12.5px]"}>
                    {f.posted_in !== null ? (
                      <RunLink token={token} n={f.posted_in} />
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className={td + " whitespace-nowrap text-[12.5px]"}>
                    {f.fixed ? (
                      <>
                        <span className="text-online">{t("pr.fixed")}</span>
                        {f.fixed_in !== null && (
                          <div>
                            <RunLink token={token} n={f.fixed_in} />
                          </div>
                        )}
                      </>
                    ) : (
                      <span className="text-ochre">{t("pr.open")}</span>
                    )}
                    {f.replied && (
                      <div className="text-faint">{t("run.replied")}</div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Note>{t("pr.findingsNote2")}</Note>
    </Card>
  );
}

/** CallsCard lists the PR's replies and Q&A answers: no run and no page
 *  each, but money the PR total and its cap count. */
function CallsCard({ d, pr }: { d: PRPageData; pr: string | null }) {
  const { t } = useTranslation();
  if (d.calls_failed) {
    return (
      <Card title={t("pr.callsTitle")}>
        <Empty>{t("pr.callsFailed")}</Empty>
      </Card>
    );
  }
  return (
    <Card title={t("pr.callsTitle")} note={t("pr.callsSub")}>
      <div className="overflow-x-auto overscroll-x-contain">
        <table className={table}>
          <thead>
            <tr>
              <th className={th}>{t("pr.when")}</th>
              <th className={th}>{t("pr.callKind")}</th>
              <th className={th}>{t("pr.model")}</th>
              <th className={thNum}>{t("run.cost")}</th>
            </tr>
          </thead>
          <tbody>
            {d.calls.map((c: CommentCall, i) => {
              const href = commentUrl(pr, c);
              const kind = known("callKind", c.kind)
                ? t(`callKind.${c.kind}`)
                : c.kind;
              return (
                <tr key={i}>
                  <td
                    className={
                      td +
                      " font-mono text-[13px] whitespace-nowrap text-muted tabular-nums"
                    }
                  >
                    {when(c.created_at)}
                  </td>
                  <td className={td}>
                    {href ? (
                      <a
                        href={href}
                        target="_blank"
                        rel="noreferrer"
                        className="hover:underline"
                      >
                        {kind} ↗
                      </a>
                    ) : (
                      kind
                    )}
                  </td>
                  <td className={tdTag + " text-muted"}>{c.model}</td>
                  <td className={tdNum}>
                    {c.cost_usd === null ? "—" : money(c.cost_usd)}
                  </td>
                </tr>
              );
            })}
            {d.more_calls && (
              <tr>
                <td
                  className={td + " text-[12.5px] text-faint italic"}
                  colSpan={4}
                >
                  {t("pr.moreCalls")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** PathText is a path with its directory faint; a path with no name after
 *  its last slash prints whole. */
function PathText({ path }: { path: string }) {
  const cut = path.lastIndexOf("/") + 1;
  if (cut === 0 || cut === path.length) return <>{path}</>;
  return (
    <>
      <span className="text-faint">{path.slice(0, cut)}</span>
      {path.slice(cut)}
    </>
  );
}

function FilesTab({ d, token }: { d: PRPageData; token: string }) {
  const { t } = useTranslation();
  if (d.files.length === 0) {
    return (
      <Card>
        <Empty>{t("run.noFiles")}</Empty>
      </Card>
    );
  }
  return (
    <Card title={t("pr.filesTitle")}>
      <div className="overflow-x-auto overscroll-x-contain">
        <table className={table}>
          <thead>
            <tr>
              <th className={th}>{t("run.file")}</th>
              <th className={th}>{t("pr.reviewedIn")}</th>
              <th className={thNum}>{t("pr.findings")}</th>
            </tr>
          </thead>
          <tbody>
            {d.files.map((f) => (
              <tr key={f.path}>
                {/* The whole path, faint up to the name: across all of a
                    PR's runs two files sharing a name are likely. */}
                <td className={td + " font-mono text-[12.5px] break-all"}>
                  <PathText path={f.path} />
                </td>
                <td className={td + " text-[12.5px]"}>
                  <span className="flex flex-wrap gap-x-3 gap-y-0.5">
                    {f.runs.map((r) => (
                      <span key={r.n} className="whitespace-nowrap">
                        <RunLink token={token} n={r.n} />{" "}
                        <span className="text-faint">
                          {known("sent", r.sent) ? t(`sent.${r.sent}`) : r.sent}
                        </span>
                      </span>
                    ))}
                  </span>
                </td>
                <td className={tdNum}>{f.findings || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** CostTab is the PR total against its cap, each priced run on one scale,
 *  and the tokens summed over the runs. Replies and answers are in the
 *  total and listed with their cost in their own view: never re-added
 *  here, in floats, from a list that may be cut. */
function CostTab({ d }: { d: PRPageData }) {
  const { t } = useTranslation();
  const rows = d.runs
    .filter((r) => r.cost_usd !== null)
    .map((r) => ({
      k: t("run.crumb", { n: r.n }),
      v: Number(r.cost_usd),
      usd: r.cost_usd,
    }));
  const scale = Math.max(0, ...rows.map((r) => r.v));
  const cost = d.cost_usd !== null ? Number(d.cost_usd) : null;
  const cap = d.cap_usd !== null ? Number(d.cap_usd) : null;
  const uncached = d.prompt_tokens - d.cached_tokens;
  return (
    <>
      <Card
        title={t("pr.costTitle")}
        // The cap row below carries the total; without a cap, the header.
        note={cap !== null && cap > 0 ? undefined : money(d.cost_usd)}
      >
        {rows.length === 0 ? (
          <Empty>{t("pr.noCost")}</Empty>
        ) : (
          rows.map((r, i) => (
            <div
              key={i}
              className="mb-1.5 grid grid-cols-[minmax(80px,160px)_1fr_64px] items-center gap-3 text-[13px]"
            >
              <span className="truncate text-ink-dim">{r.k}</span>
              <div className="h-3.5">
                {scale > 0 && r.v > 0 && (
                  <div
                    className="h-full rounded-[3px] bg-s1"
                    style={{ width: `${Math.max(0.5, (r.v / scale) * 100)}%` }}
                  />
                )}
              </div>
              <span className="text-right font-mono text-[12.5px] text-muted tabular-nums">
                {money(r.usd)}
              </span>
            </div>
          ))
        )}
        {cost !== null && cap !== null && cap > 0 && (
          <div className="mt-4 grid grid-cols-[80px_1fr_150px] items-center gap-3 text-[13px]">
            <span className="text-faint">{t("run.prCap")}</span>
            <CapMeter cost={cost} cap={cap} className="h-2.5 rounded-[5px]" />

            <span className="text-right font-mono text-[12.5px] text-ink-dim">
              {t("run.prOf", {
                cost: money(d.cost_usd),
                cap: money(d.cap_usd),
              })}
            </span>
          </div>
        )}
        <Note>{t("pr.costNote")}</Note>
      </Card>
      <Card title={t("pr.tokensTitle")}>
        <div className="overflow-x-auto overscroll-x-contain">
          <table className={table}>
            <tbody>
              <tr>
                <td className={td}>{t("run.laneInput")}</td>
                <td className={tdNum}>{uncached.toLocaleString()}</td>
              </tr>
              <tr>
                <td className={td}>{t("run.laneCacheRead")}</td>
                <td className={tdNum}>{d.cached_tokens.toLocaleString()}</td>
              </tr>
              <tr>
                <td className={td}>{t("run.laneOutput")}</td>
                <td className={tdNum}>
                  {d.completion_tokens.toLocaleString()}
                </td>
              </tr>
              <tr>
                <td className={td + " font-medium text-ink"}>
                  {t("run.total")}
                </td>
                <td className={tdNum + " text-ink"}>
                  {(d.prompt_tokens + d.completion_tokens).toLocaleString()}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
