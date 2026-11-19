// The run page's views of the review's own record (RunPipeline): where the
// time went stage by stage, the context the model had, what the prompt was
// made of, how each file was sent, and the findings that were not posted.
// Headlines only, like everything on the page.
import { useTranslation } from "react-i18next";
import type { RunPipeline } from "./api";
import {
  duration,
  known,
  num,
  severityTone,
  severityWord,
  sourceUrl,
} from "./format";
import {
  Card,
  Empty,
  KV,
  Pill,
  table,
  td,
  tdNum,
  tdTag,
  th,
  thNum,
} from "./ui";

// A stage the page does not know (a newer binary) still draws, under its
// own name. Waits are hatched, the gateway call has its own hue, the rest is
// noergler and Bitbucket. review.StageSplit counts the same set as waits for
// the PR page: change both.
const WAITS = new Set(["queued", "to_post"]);

export function StageTimeline({ p }: { p: RunPipeline }) {
  const { t } = useTranslation();
  const total = p.stages.reduce((s, x) => s + x.ms, 0);
  if (p.stages.length === 0 || total <= 0) return null;
  // Each stage starts where the ones before it ended.
  const starts = p.stages.map((_, i) =>
    p.stages.slice(0, i).reduce((s, x) => s + x.ms, 0),
  );
  return (
    <>
      {p.stages.map((s, i) => {
        const left = (starts[i] / total) * 100;
        const width = Math.max(0.4, (s.ms / total) * 100);
        const label = known("stage", s.name) ? t(`stage.${s.name}`) : s.name;
        const cls = WAITS.has(s.name)
          ? "texture-wait"
          : s.name === "gateway"
            ? "bg-s3"
            : "bg-s1";
        return (
          <div
            key={`${s.name}-${i}`}
            className="mb-1.5 grid grid-cols-[minmax(110px,200px)_1fr_64px] items-center gap-3 text-[13px]"
          >
            <span
              className={
                "truncate " +
                (WAITS.has(s.name) ? "text-faint" : "text-ink-dim")
              }
            >
              {label}
            </span>
            <div className="relative h-3.5">
              <div className="absolute inset-x-0 top-1.5 h-px bg-border-soft" />
              <div
                className={"absolute inset-y-0 rounded-[3px] " + cls}
                style={{ left: `${left}%`, width: `${width}%` }}
                title={`${label}: ${duration(s.ms)}`}
              />
            </div>
            <span className="text-right font-mono text-[12.5px] text-muted tabular-nums">
              {duration(s.ms)}
            </span>
          </div>
        );
      })}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-muted">
        <span className="inline-flex items-center gap-1.5">
          <i
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-[2px] texture-wait"
          />
          {t("run.legendWait")}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-[2px] bg-s1"
          />
          {t("run.timeNoergler")}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-[2px] bg-s3"
          />
          {t("run.timeGateway")}
        </span>
      </div>
    </>
  );
}

export function ContextCard({ p }: { p: RunPipeline }) {
  const { t } = useTranslation();
  const a = p.agents_md;
  return (
    <Card title={t("run.contextTitle")}>
      <dl className="grid grid-cols-1 gap-x-7 sm:grid-cols-2">
        <KV
          k={t("promptPart.repo_instructions")}
          v={
            !a.found
              ? t("run.agentsMissing")
              : a.warn_tokens > 0
                ? t("run.agentsOf", {
                    tokens: num(a.tokens),
                    warn: num(a.warn_tokens),
                    pct: Math.round((a.tokens / a.warn_tokens) * 100),
                  })
                : t("run.agentsTokens", { tokens: num(a.tokens) })
          }
        />
        <KV k={t("run.ticket")} mono v={p.ticket || "—"} />
        {p.parent_ticket && (
          <KV k={t("run.parentTicket")} mono v={p.parent_ticket} />
        )}
        <KV
          k={t("run.earlierFindings")}
          v={t("run.shownOf", {
            shown: p.previously_posted.shown,
            total: p.previously_posted.total,
          })}
        />
        <KV
          k={t("run.crossFile")}
          mono
          v={p.cross_file.length === 0 ? "—" : p.cross_file.join(", ")}
        />
        {p.http.map((h) => (
          <KV
            key={h.upstream}
            k={t("run.httpCalls", { upstream: h.upstream })}
            mono
            v={num(h.calls)}
          />
        ))}
      </dl>
    </Card>
  );
}

/** The prompt's variable parts in the order review.txt carries them: the
 *  stable context, then the files, then what changes run to run. The
 *  template's own text runs between them all and is listed first as one
 *  sum; compliance instructions are in the rest. */
const PARTS = [
  "template",
  "repo_instructions",
  "ticket",
  "pr_description",
  "files",
  "cumulative_diff",
  "previously_posted",
] as const;

export function InputTab({ p }: { p: RunPipeline | null }) {
  const { t } = useTranslation();
  if (p === null) {
    return (
      <Card>
        <Empty>{t("run.noPipeline")}</Empty>
      </Card>
    );
  }
  const max = Math.max(...PARTS.map((k) => p.prompt[k]), 1);
  const rest = p.prompt.total - PARTS.reduce((s, k) => s + p.prompt[k], 0);
  return (
    <>
      <Card
        title={t("run.promptTitle")}
        note={t("run.promptNote", {
          total: num(p.prompt.total),
          budget: num(p.prompt.budget),
        })}
      >
        <div className="overflow-x-auto overscroll-x-contain">
          <table className={table}>
            <thead>
              <tr>
                <th className={th}>{t("run.part")}</th>
                <th className={thNum}>{t("run.tokens")}</th>
                <th className={thNum}>{t("run.share")}</th>
                <th className={th + " w-[38%]"} />
              </tr>
            </thead>
            <tbody>
              {PARTS.map((k) => (
                <tr key={k}>
                  <td className={td}>{t(`promptPart.${k}`)}</td>
                  <td className={tdNum}>{num(p.prompt[k])}</td>
                  <td className={tdNum}>
                    {p.prompt.total > 0
                      ? `${Math.round((p.prompt[k] / p.prompt.total) * 100)}%`
                      : "—"}
                  </td>
                  <td className={td}>
                    <div className="h-2 overflow-hidden rounded-[4px] bg-track">
                      <div
                        className="h-full bg-s1"
                        style={{ width: `${(p.prompt[k] / max) * 100}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
              {rest > 0 && (
                <tr>
                  <td className={td + " text-muted"}>{t("promptPart.rest")}</td>
                  <td className={tdNum}>{num(rest)}</td>
                  <td className={tdNum} />
                  <td className={td} />
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title={t("run.filesTitle")}>
        {p.files.length === 0 ? (
          <Empty>{t("run.noFiles")}</Empty>
        ) : (
          <div className="overflow-x-auto overscroll-x-contain">
            <table className={table}>
              <thead>
                <tr>
                  <th className={th}>{t("run.file")}</th>
                  <th className={th}>{t("run.howSent")}</th>
                  <th className={thNum}>+</th>
                  <th className={thNum}>−</th>
                  <th className={thNum}>{t("run.tokens")}</th>
                </tr>
              </thead>
              <tbody>
                {p.files.map((f) => {
                  const out = [
                    "oversized",
                    "over_budget",
                    "not_reviewable",
                  ].includes(f.sent);
                  return (
                    <tr key={f.path}>
                      <td
                        className={tdTag + (out ? " text-faint" : "")}
                        title={f.path}
                      >
                        <span className="block max-w-[24ch] truncate">
                          {f.path.split("/").pop() || f.path}
                        </span>
                      </td>
                      <td className={td + (out ? " text-faint" : "")}>
                        {known("sent", f.sent) ? t(`sent.${f.sent}`) : f.sent}
                      </td>
                      <td className={tdNum}>{num(f.added)}</td>
                      <td className={tdNum}>{num(f.removed)}</td>
                      <td className={tdNum}>
                        {f.tokens > 0 ? num(f.tokens) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}

const prose = "mt-1 whitespace-pre-wrap break-words text-[13px] text-muted";

/** FindingText is a finding's headline over its full text: the comment
 *  (or explanation), the suggestion, the author's dispute and the quoted
 *  lines. Plain text, never rendered markdown: it is the model's and the
 *  author's. One renderer for every findings table on the run page. */
export function FindingText({
  headline,
  comment,
  suggestion,
  dispute,
  evidence,
}: {
  headline: string;
  comment?: string | null;
  suggestion?: string | null;
  dispute?: string | null;
  evidence?: string[];
}) {
  const { t } = useTranslation();
  return (
    <>
      <div className="text-ink">{headline || "—"}</div>
      {comment && <div className={prose}>{comment}</div>}
      {suggestion && (
        <div className={prose}>
          <span className="text-faint">{t("run.suggestion")}: </span>
          {suggestion}
        </div>
      )}
      {dispute && (
        <div className={prose}>
          <span className="text-faint">{t("run.dispute")}: </span>
          {dispute}
        </div>
      )}
      <Evidence lines={evidence} />
    </>
  );
}

/** Evidence is the lines the model quoted, as code. */
function Evidence({ lines }: { lines?: string[] }) {
  if (!lines || lines.length === 0) return null;
  return (
    <pre className="mt-1.5 overflow-x-auto rounded-ui-sm border border-border-soft px-2 py-1 font-mono text-[12px] text-ink-dim">
      {lines.join("\n")}
    </pre>
  );
}

export function NotPostedCard({ p }: { p: RunPipeline }) {
  const { t } = useTranslation();
  if (p.decisions.length === 0) return null;
  return (
    <Card title={t("run.notPostedTitle")}>
      <div className="overflow-x-auto overscroll-x-contain">
        <table className={table}>
          <thead>
            <tr>
              <th className={th + " w-px"}>{t("run.severity")}</th>
              <th className={th + " w-px"}>{t("run.where")}</th>
              <th className={th}>{t("run.finding")}</th>
              <th className={th + " w-px"}>{t("run.result")}</th>
            </tr>
          </thead>
          <tbody>
            {p.decisions.map((f, i) => (
              <tr key={`${f.file}:${f.line}:${i}`}>
                <td className={td}>
                  <Pill tone={severityTone(f.severity)}>
                    {severityWord(f.severity)}
                  </Pill>
                </td>
                <td className={tdTag + " text-muted"}>
                  {f.file}:{f.line}
                </td>
                <td className={td}>
                  <FindingText
                    headline={f.headline}
                    comment={f.comment}
                    suggestion={f.suggestion}
                    evidence={f.evidence}
                  />
                </td>
                <td className={td + " whitespace-nowrap text-[12.5px]"}>
                  <span
                    className={
                      f.decision === "repeat" ? "text-muted" : "text-danger-ink"
                    }
                  >
                    {known("decision", f.decision)
                      ? t(`decision.${f.decision}`)
                      : f.decision}
                  </span>
                  {f.reason && (
                    <div className="text-[11.5px] text-faint">
                      {known("dropReason", f.reason)
                        ? t(`dropReason.${f.reason}`)
                        : f.reason}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** PreExistingCard lists the bugs the model saw in code this PR did not
 *  change. Never posted to the PR, so this is the only place they are said:
 *  the explanation, the quoted lines and a link to the line. */
export function PreExistingCard({
  p,
  tag,
  base,
}: {
  p: RunPipeline;
  tag: string;
  base: string;
}) {
  const { t } = useTranslation();
  const bugs = p.pre_existing ?? [];
  if (bugs.length === 0) return null;
  return (
    <Card title={t("run.preExistingTitle")}>
      {/* The same table as the posted findings: where, linked to the line
          at the reviewed commit, and the finding in full. */}
      <div className="overflow-x-auto overscroll-x-contain">
        <table className={table}>
          <thead>
            <tr>
              <th className={th + " w-px"}>{t("run.where")}</th>
              <th className={th}>{t("run.finding")}</th>
            </tr>
          </thead>
          <tbody>
            {bugs.map((b, i) => {
              const href = sourceUrl(tag, base, b.file, b.line, p.commit);
              const where = `${b.file}:${b.line}`;
              return (
                <tr key={`${where}:${i}`}>
                  <td className={tdTag}>
                    {href ? (
                      <a
                        href={href}
                        target="_blank"
                        rel="noreferrer"
                        className="hover:text-ink hover:underline"
                      >
                        {where} ↗
                      </a>
                    ) : (
                      where
                    )}
                  </td>
                  <td className={td}>
                    <FindingText
                      headline={b.headline}
                      comment={b.explanation}
                      evidence={b.evidence}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** ComplianceCard lists every ticket requirement the model named and what
 *  the summary did with it: shown, with the line that proves it, or left
 *  out and why. The summary never says a requirement is unmet (a PR is a
 *  delta), so a requirement the model called unmet is only seen here. */
export function ComplianceCard({ p }: { p: RunPipeline }) {
  const { t } = useTranslation();
  const reqs = p.compliance ?? [];
  if (reqs.length === 0) return null;
  return (
    <Card title={t("run.complianceTitle")}>
      <div className="overflow-x-auto overscroll-x-contain">
        <table className={table}>
          <thead>
            <tr>
              <th className={th}>{t("run.requirement")}</th>
              <th className={th + " w-px"}>{t("run.result")}</th>
            </tr>
          </thead>
          <tbody>
            {reqs.map((r, i) => (
              <tr key={`${r.requirement}:${i}`}>
                <td className={td}>
                  <div className="text-ink">{r.requirement || "—"}</div>
                  {r.file && (
                    <div className="mt-1 font-mono text-[12px] text-muted">
                      {r.file}
                    </div>
                  )}
                  <Evidence lines={r.quote ? [r.quote] : undefined} />
                </td>
                <td className={td + " whitespace-nowrap text-[12.5px]"}>
                  <Pill tone={r.drop ? "quiet" : "ok"}>
                    {r.drop ? t("run.leftOut") : t("run.shown")}
                  </Pill>
                  {r.drop && (
                    <div className="mt-1 text-[11.5px] text-faint">
                      {known("complianceDrop", r.drop)
                        ? t(`complianceDrop.${r.drop}`)
                        : r.drop}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
