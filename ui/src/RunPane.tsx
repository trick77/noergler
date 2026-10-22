// The run pane: one attempt of a PR and everything recorded about it, in
// tabs so the summary is never below a page of detail. Shown inside the
// PR's shell (PRPage), beside the rail of its runs, so the PR's title,
// history, cost and calls are the shell's; the pane starts at the run's
// own line. Reached only through the PR's summary comment, so it shows the
// full text.
import { useEffect, useState, type KeyboardEvent, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { RunCall, RunDetail, RunFinding } from "./api";
import {
  commentUrl,
  duration,
  money,
  outcomeTitle,
  outcomeTone,
  outcomeWord,
  prUrl,
  severityTone,
  severityWord,
} from "./format";
import {
  ContextCard,
  FindingText,
  InputTab,
  NotPostedCard,
  PreExistingCard,
  StageTimeline,
} from "./RunPipeline";
import { usePoll } from "./usePoll";
import {
  Card,
  Empty,
  Failed,
  Note,
  Pill,
  Sep,
  Tile,
  Tiles,
  table,
  td,
  tdNum,
  tdTag,
  th,
  thNum,
} from "./ui";

type Tab = "overview" | "findings" | "cost" | "input";
const TABS: Tab[] = ["overview", "findings", "cost", "input"];

/** The tab lives in the hash (#findings), so a link can open one and a
 *  reload keeps it. */
function tabFromHash(): Tab {
  const h = window.location.hash.slice(1);
  return (TABS as string[]).includes(h) ? (h as Tab) : "overview";
}

function num(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toLocaleString();
}

export function RunPane({ token, n }: { token: string; n: number }) {
  const { t } = useTranslation();
  // A finished run does not change; the slow poll only picks up a run that
  // was still being recorded when the page opened.
  const { data, failed } = usePoll<RunDetail>(`prs/${token}/runs/${n}`, 60000);
  const [tab, setTab] = useState<Tab>(tabFromHash);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const pick = (next: Tab) => {
    setTab(next);
    window.history.replaceState(
      {},
      "",
      next === "overview" ? window.location.pathname : `#${next}`,
    );
  };

  if (!data) {
    return failed ? (
      <Failed what={t("run.failedWhat")} />
    ) : (
      <div className="skeleton h-24 rounded-ui" />
    );
  }

  const d = data;
  const pr = prUrl(d.tag, d.bitbucket_url);
  const call = d.call;
  const inTokens = call?.input_tokens ?? d.run?.prompt_tokens ?? null;
  const outTokens = call?.output_tokens ?? d.run?.completion_tokens ?? null;
  const cached = call?.cache_read_tokens ?? d.run?.cached_tokens ?? null;

  const counts: Record<Tab, ReactNode> = {
    overview: null,
    findings: d.run ? d.findings.length : null,
    cost: null,
    input: d.detail ? d.detail.files.length : null,
  };
  const labels: Record<Tab, string> = {
    overview: t("run.tabOverview"),
    findings: t("run.tabFindings"),
    cost: t("run.tabCost"),
    input: t("run.tabInput"),
  };

  const onTabKey = (e: KeyboardEvent, i: number) => {
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (step === 0) return;
    e.preventDefault();
    const n = TABS[(i + step + TABS.length) % TABS.length];
    pick(n);
    document.getElementById(`run-tab-${n}`)?.focus();
  };

  return (
    <>
      <div className="flex flex-wrap items-baseline gap-x-3.5 gap-y-1">
        <h3 className="font-serif text-[20px] leading-tight font-medium tracking-tight text-ink">
          {t("run.crumb", { n: d.n })}
        </h3>
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 text-[13.5px] text-muted">
          <Pill
            tone={outcomeTone(d.outcome)}
            title={outcomeTitle(d.outcome, d.reason, d.reason_label)}
          >
            {outcomeWord(d.outcome, d.reason)}
          </Pill>
          <span>
            {d.kind === "mention" ? t("run.mention") : t("run.auto")}
            {d.run && (
              <>
                <Sep />
                {d.run.incremental ? t("run.incremental") : t("run.full")}
                {d.run.to_commit && (
                  <span className="ml-1.5 font-mono text-[12.5px]">
                    {d.run.from_commit
                      ? `${d.run.from_commit.slice(0, 7)} → `
                      : ""}
                    {d.run.to_commit.slice(0, 7)}
                  </span>
                )}
              </>
            )}
          </span>
          <Sep />
          <span>
            {new Date(d.created_at).toLocaleString(undefined, {
              dateStyle: "medium",
              timeStyle: "medium",
            })}
          </span>
        </div>
      </div>

      <div className="mt-5">
        <Tiles>
          <Tile label={t("run.cost")} value={money(d.cost_usd ?? null)} />
          <Tile label={t("run.elapsed")} value={duration(d.elapsed_ms)}>
            {call?.headers_ms != null && (
              <div className="mt-0.5 text-[12px] text-faint">
                {t("run.inModel", { time: duration(call.headers_ms) })}
              </div>
            )}
          </Tile>
          <Tile
            label={t("run.findings")}
            value={d.run ? d.run.findings_posted : "—"}
          >
            {d.fixed.length > 0 && (
              <div className="mt-0.5 text-[12px] text-faint">
                {t("run.fixedEarlier", { count: d.fixed.length })}
              </div>
            )}
          </Tile>
          <Tile
            label={t("run.tokens")}
            value={
              inTokens === null && outTokens === null
                ? "—"
                : ((inTokens ?? 0) + (outTokens ?? 0)).toLocaleString()
            }
          >
            {inTokens ? (
              <div className="mt-0.5 text-[12px] text-faint">
                {t("run.fromCache", {
                  pct: Math.round(((cached ?? 0) / inTokens) * 100),
                })}
              </div>
            ) : null}
          </Tile>
        </Tiles>
      </div>

      <div
        role="tablist"
        aria-label={t("run.tabsLabel")}
        className="sticky top-0 z-10 mt-6 mb-4 flex gap-0.5 overflow-x-auto border-b border-border bg-bg [scrollbar-width:none]"
      >
        {TABS.map((k, i) => (
          <button
            key={k}
            id={`run-tab-${k}`}
            type="button"
            role="tab"
            aria-selected={tab === k}
            aria-controls={`run-panel-${k}`}
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
            {labels[k]}
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
        id={`run-panel-${tab}`}
        aria-labelledby={`run-tab-${tab}`}
      >
        {tab === "overview" && <Overview d={d} />}
        {tab === "findings" && <Findings d={d} pr={pr} />}
        {tab === "cost" && <CostTab call={call} />}
        {tab === "input" && <InputTab p={d.detail} />}
      </div>
    </>
  );
}

/** KV is a label/value pair in the two-column key list. */
function KV({ k, v, mono }: { k: string; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex min-w-0 justify-between gap-3 border-b border-border-soft py-1.5 text-[13.5px]">
      <dt className="whitespace-nowrap text-muted">{k}</dt>
      <dd
        className={
          "min-w-0 text-right break-words text-ink-dim " +
          (mono ? "font-mono text-[12.5px]" : "")
        }
      >
        {v}
      </dd>
    </div>
  );
}

const kvGrid = "grid grid-cols-1 gap-x-7 sm:grid-cols-2";

function Overview({ d }: { d: RunDetail }) {
  const { t } = useTranslation();
  const total = d.elapsed_ms;
  const gw = d.call?.total_ms ?? null;
  const stages = d.detail?.stages ?? [];
  // The stages run past the elapsed time (recording comes after it), so a
  // timeline is captioned with its own sum.
  const stagesMs = stages.reduce((s, x) => s + x.ms, 0);

  if (d.outcome === "skipped") {
    return (
      <Card title={t("run.skippedTitle")}>
        <p className="text-[14px] text-ink-dim">
          {outcomeTitle(d.outcome, d.reason, d.reason_label)}
        </p>
      </Card>
    );
  }

  return (
    <>
      <Card
        title={t("run.timeTitle")}
        note={
          d.detail && stagesMs > 0
            ? duration(stagesMs)
            : total !== null
              ? t("run.timeNote", { time: duration(total) })
              : undefined
        }
      >
        {d.detail && stagesMs > 0 ? (
          <StageTimeline p={d.detail} />
        ) : total !== null && gw !== null && gw <= total ? (
          <>
            <TimeBar
              label={t("run.timeNoergler")}
              ms={total - gw}
              total={total}
              cls="bg-s1"
            />
            <TimeBar
              label={t("run.timeGateway")}
              ms={gw}
              total={total}
              cls="bg-s3"
            />
          </>
        ) : (
          <Empty>{t("run.noCallTime")}</Empty>
        )}
      </Card>
      {d.detail && <ContextCard p={d.detail} />}
      {d.run && (
        <Card title={t("run.factsTitle")}>
          <dl className={kvGrid}>
            <KV
              k={t("run.mode")}
              v={d.run.incremental ? t("run.incremental") : t("run.full")}
            />
            <KV
              k={t("run.commits")}
              mono
              v={
                d.run.to_commit
                  ? `${d.run.from_commit ? d.run.from_commit.slice(0, 7) + " → " : ""}${d.run.to_commit.slice(0, 7)}`
                  : "—"
              }
            />
            <KV k={t("run.profile")} mono v={d.run.model} />
            <KV k={t("run.filesChanged")} v={num(d.run.files_changed)} />
            <KV
              k={t("run.lines")}
              mono
              v={`+${d.run.lines_added} / −${d.run.lines_removed}`}
            />
          </dl>
        </Card>
      )}
    </>
  );
}

function TimeBar({
  label,
  ms,
  total,
  cls,
}: {
  label: string;
  ms: number;
  total: number;
  cls: string;
}) {
  const pct = total > 0 ? Math.max(0.5, (ms / total) * 100) : 0;
  return (
    <div className="mb-2 grid grid-cols-[minmax(110px,190px)_1fr_64px] items-center gap-3 text-[13px]">
      <span className="truncate text-ink-dim">{label}</span>
      <div className="relative h-3.5 rounded-[3px] bg-border-soft">
        <div
          className={"absolute inset-y-0 left-0 rounded-[3px] " + cls}
          style={{ width: `${pct}%` }}
          title={`${label}: ${duration(ms)}`}
        />
      </div>
      <span className="text-right font-mono text-[12.5px] text-muted tabular-nums">
        {duration(ms)}
      </span>
    </div>
  );
}

function FindingRows({
  rows,
  pr,
  fixedTable,
  moved = [],
}: {
  rows: RunFinding[];
  pr: string | null;
  fixedTable?: boolean;
  /** Validation's moves, matched to a posted finding by file and new line. */
  moved?: { file: string; from: number; to: number }[];
}) {
  const { t } = useTranslation();
  return (
    <div className="overflow-x-auto overscroll-x-contain">
      <table className={table}>
        <thead>
          <tr>
            <th className={th + " w-px"}>{t("run.severity")}</th>
            <th className={th + " w-px"}>{t("run.where")}</th>
            <th className={th}>{t("run.finding")}</th>
            {!fixedTable && <th className={th + " w-px"}>{t("run.result")}</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((f, i) => {
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
                {/* The headline only: the full text is one click away on
                    the PR, through the location link. */}
                <td className={td}>
                  <FindingText headline={f.headline} />
                </td>
                {/* The fixed table has no result column: every row is
                    fixed. Which run posted it is the PR page's. */}
                {!fixedTable && (
                  <td className={td + " whitespace-nowrap text-[12.5px]"}>
                    <>
                      <span className="text-online">{t("run.posted")}</span>
                      {f.replied && (
                        <div className="text-faint">{t("run.replied")}</div>
                      )}
                      {f.fixed && (
                        <div className="text-faint">{t("run.fixedLater")}</div>
                      )}
                      {moved
                        .filter((m) => m.file === f.file && m.to === f.line)
                        .map((m, i) => (
                          <div key={i} className="text-faint">
                            {t("run.movedFrom", { line: m.from })}
                          </div>
                        ))}
                    </>
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Findings({ d, pr }: { d: RunDetail; pr: string | null }) {
  const { t } = useTranslation();
  if (!d.run) {
    return (
      <Card>
        <Empty>{t("run.noRun")}</Empty>
      </Card>
    );
  }
  return (
    <>
      <Card title={t("run.tabFindings")}>
        {d.findings.length === 0 ? (
          <Empty>{t("run.noFindings")}</Empty>
        ) : (
          <FindingRows
            rows={d.findings}
            pr={pr}
            moved={d.detail?.moved ?? []}
          />
        )}
        {d.detail?.verdict &&
          d.detail.verdict.model !== d.detail.verdict.posted && (
            <Note>
              {t("run.verdictLowered", {
                model: d.detail.verdict.model,
                posted: d.detail.verdict.posted,
              })}
            </Note>
          )}
      </Card>
      {d.detail && <NotPostedCard p={d.detail} />}
      {d.detail && (
        <PreExistingCard p={d.detail} tag={d.tag} base={d.bitbucket_url} />
      )}
      {d.fixed.length > 0 && (
        <Card title={t("run.fixedTitle")}>
          <FindingRows rows={d.fixed} pr={pr} fixedTable />
        </Card>
      )}
    </>
  );
}

/** Seg is one segment of a stacked token bar; zero or absent draws nothing. */
function Seg({
  v,
  cls,
  title,
}: {
  v: number | null;
  cls: string;
  title: string;
}) {
  if (!v) return null;
  return <div className={"h-full " + cls} style={{ flex: v }} title={title} />;
}

function CostTab({ call }: { call: RunCall | null }) {
  const { t } = useTranslation();
  if (call === null) {
    return (
      <Card>
        <Empty>{t("run.noCall")}</Empty>
      </Card>
    );
  }
  const absent = t("run.absent");
  const tok = (v: number | null) => (v === null ? absent : v.toLocaleString());
  const lane = (v: string | null) => (v === null ? absent : money(v));
  const input = call.input_tokens ?? 0;
  const output = call.output_tokens ?? 0;
  const scale = Math.max(input, output, 1);
  const reasoning = call.reasoning_tokens;
  const outText =
    call.output_tokens !== null && reasoning !== null
      ? call.output_tokens - reasoning
      : call.output_tokens;
  const totalCost = call.cost_usd !== null ? Number(call.cost_usd) : null;
  const share = (v: string | null) =>
    v === null || !totalCost ? null : (Number(v) / totalCost) * 100;

  const lanes: {
    k: string;
    tokens: number | null;
    v: string | null;
    sub?: boolean;
  }[] = [
    {
      k: t("run.laneInput"),
      tokens: call.input_uncached_tokens,
      v: call.cost_input_usd,
    },
    {
      k: t("run.laneCacheRead"),
      tokens: call.cache_read_tokens,
      v: call.cost_cache_read_usd,
    },
    {
      k: t("run.laneCacheWrite"),
      tokens: call.cache_write_tokens,
      v: call.cost_cache_write_usd,
    },
    {
      k: t("run.laneOutput"),
      tokens: call.output_tokens,
      v: call.cost_output_usd,
    },
    {
      k: t("run.laneReasoning"),
      tokens: reasoning,
      v: call.cost_reasoning_usd,
      sub: true,
    },
    { k: t("run.laneTool"), tokens: null, v: call.cost_tool_usage_usd },
  ];

  return (
    <>
      <Card
        title={t("run.tokensTitle")}
        note={t("run.tokensTotal", {
          total: (input + output).toLocaleString(),
        })}
      >
        {[
          {
            k: t("run.input"),
            total: input,
            segs: [
              {
                v: call.input_uncached_tokens,
                cls: "bg-s1",
                n: t("run.uncached"),
              },
              {
                v: call.cache_read_tokens,
                cls: "bg-s2",
                n: t("run.cacheRead"),
              },
              {
                v: call.cache_write_tokens,
                cls: "bg-s3",
                n: t("run.cacheWrite"),
              },
            ],
          },
          {
            k: t("run.output"),
            total: output,
            segs: [
              { v: outText, cls: "bg-s4", n: t("run.outputText") },
              { v: reasoning, cls: "texture-s4", n: t("run.reasoning") },
            ],
          },
        ].map((row) => (
          <div
            key={row.k}
            className="mb-2.5 grid grid-cols-[64px_1fr_86px] items-center gap-3 text-[13px]"
          >
            <span className="text-faint">{row.k}</span>
            <div className="h-5">
              <div
                className="flex h-full gap-0.5 overflow-hidden rounded-[4px]"
                style={{ width: `${(row.total / scale) * 100}%` }}
              >
                {row.segs.map((s) => (
                  <Seg
                    key={s.n}
                    v={s.v}
                    cls={s.cls}
                    title={`${s.n}: ${tok(s.v)}`}
                  />
                ))}
              </div>
            </div>
            <span className="text-right font-mono text-[12.5px] text-ink-dim tabular-nums">
              {row.total.toLocaleString()}
            </span>
          </div>
        ))}
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-muted">
          {[
            {
              cls: "bg-s1",
              n: t("run.uncached"),
              v: call.input_uncached_tokens,
            },
            { cls: "bg-s2", n: t("run.cacheRead"), v: call.cache_read_tokens },
            {
              cls: "bg-s3",
              n: t("run.cacheWrite"),
              v: call.cache_write_tokens,
            },
            { cls: "bg-s4", n: t("run.outputText"), v: outText },
            { cls: "texture-s4", n: t("run.reasoning"), v: reasoning },
          ].map((l) => (
            <span key={l.n} className="inline-flex items-center gap-1.5">
              <i
                aria-hidden
                className={"inline-block h-2.5 w-2.5 rounded-[2px] " + l.cls}
              />
              {l.n}
              <span
                className={
                  "font-mono text-[12px] " +
                  (l.v === null ? "text-faint italic" : "text-ink-dim")
                }
              >
                {tok(l.v)}
              </span>
            </span>
          ))}
        </div>
      </Card>

      <Card title={t("run.costTitle")}>
        <div className="overflow-x-auto overscroll-x-contain">
          <table className={table}>
            <thead>
              <tr>
                <th className={th}>{t("run.lane")}</th>
                <th className={thNum}>{t("run.tokens")}</th>
                <th className={thNum}>{t("run.usd")}</th>
                <th className={th + " w-[34%]"}>{t("run.share")}</th>
              </tr>
            </thead>
            <tbody>
              {lanes.map((l) => {
                const pct = l.sub ? null : share(l.v);
                return (
                  <tr key={l.k}>
                    <td
                      className={
                        td + (l.sub ? " pl-4 text-[12.5px] text-muted" : "")
                      }
                    >
                      {l.k}
                    </td>
                    <td
                      className={
                        tdNum + (l.tokens === null ? " italic text-faint" : "")
                      }
                    >
                      {l.tokens === null
                        ? l.k === t("run.laneTool")
                          ? "—"
                          : absent
                        : l.tokens.toLocaleString()}
                    </td>
                    <td
                      className={
                        tdNum + (l.v === null ? " italic text-faint" : "")
                      }
                    >
                      {lane(l.v)}
                    </td>
                    <td className={td}>
                      {pct !== null && (
                        <div className="h-2 overflow-hidden rounded-[4px] bg-track">
                          <div
                            className="h-full bg-s1"
                            style={{ width: `${Math.min(100, pct)}%` }}
                          />
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
              <tr>
                <td className={td + " font-medium text-ink"}>
                  {t("run.total")}
                </td>
                <td className={tdNum + " text-ink"}>
                  {num(
                    call.input_tokens !== null || call.output_tokens !== null
                      ? input + output
                      : null,
                  )}
                </td>
                <td className={tdNum + " text-ink"}>
                  {money(call.cost_usd ?? null)}
                </td>
                <td className={td} />
              </tr>
            </tbody>
          </table>
        </div>
        <dl className={kvGrid + " mt-3"}>
          <KV
            k={t("run.keySpend")}
            mono
            v={call.key_spend_usd ? money(call.key_spend_usd) : absent}
          />
        </dl>
      </Card>

      <Card title={t("run.gatewayTitle")}>
        <dl className={kvGrid}>
          <KV k={t("run.profile")} mono v={call.profile} />
          <KV k={t("run.modelAnswered")} mono v={call.model || absent} />
          <KV k={t("run.deployment")} mono v={call.deployment || absent} />
          <KV
            k={t("run.reasoningSent")}
            mono
            v={call.reasoning_sent || absent}
          />
          <KV k={t("run.finishReason")} mono v={call.finish_reason || absent} />
          <KV k={t("run.callID")} mono v={call.call_id || absent} />
          <KV
            k={t("run.litellmVersion")}
            mono
            v={call.litellm_version || absent}
          />
          <KV k={t("run.headers")} mono v={duration(call.headers_ms)} />
          <KV k={t("run.totalCall")} mono v={duration(call.total_ms)} />
          <KV
            k={t("run.warnings")}
            v={
              call.warnings.length === 0
                ? t("run.none")
                : call.warnings.join("; ")
            }
          />
        </dl>
      </Card>
    </>
  );
}
