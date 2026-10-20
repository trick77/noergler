// The developer-facing page: what somebody asks when noergler did something
// surprising to THEIR pull request. Everything here is behaviour the code
// already has; nothing on this page is configuration advice for operators,
// which lives in CONFIGURATION.md.
//
// Static prose, no endpoint: none of it is per-instance data. Where a value
// is a per-team default it says so, because teams.yaml can override almost
// all of it and a flat number here would be wrong for somebody.
import { Trans, useTranslation } from "react-i18next";
import { reasonLabel } from "./format";
import { Card, Note, column, h2, lede, page } from "./ui";

function Q({ q, children }: { q: string; children: React.ReactNode }) {
  return (
    <Card title={q}>
      <div className="space-y-2.5 text-[13.5px] leading-relaxed text-ink-dim">
        {children}
      </div>
    </Card>
  );
}

/** Code is a literal a reader might type or search for. */
// children optional: <Trans> fills it from the catalog.
function Code({ children }: { children?: React.ReactNode }) {
  return (
    <code className="rounded-ui-sm bg-active px-1 py-0.5 font-mono text-[12.5px] text-ink">
      {children}
    </code>
  );
}

// The eleven pre-flight exits, as review.SkipReason codes, in the order
// prepare reaches them. The label is the reason catalog's; the why is
// faq.skipWhy.
const SKIPS = [
  "agents_md_missing",
  "agents_md_too_large",
  "not_auto_review_author",
  "ignored_author",
  "branch_opt_out",
  "pr_ignored",
  "pr_cost_cap",
  "no_reviewable_files",
  "empty_diff",
  "head_unchanged",
  "diff_too_large",
] as const;

type MarkupKey =
  "faq.whyNotA" | "faq.stopA1" | "faq.stopA3" | "faq.filesA2" | "faq.apiA1";

/** T renders a catalog entry that carries markup: <code>, <strong> and
 *  <docs> map to the page's own elements, so a translation can move them
 *  around the sentence without touching JSX. <docs>, not <link>: Trans
 *  parses <link> as a void HTML tag and drops its text. */
function T({ k }: { k: MarkupKey }) {
  return (
    <Trans
      i18nKey={k}
      components={{
        code: <Code />,
        strong: <strong />,
        docs: (
          <a
            href="/api/docs"
            className="text-ink underline hover:no-underline"
          />
        ),
      }}
    />
  );
}

export function FaqPage() {
  const { t } = useTranslation();
  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>{t("nav.faq")}</h2>
        <p className={lede}>{t("faq.lede")}</p>

        <Q q={t("faq.whyNotQ")}>
          <p>
            <T k="faq.whyNotA" />
          </p>
          <ul className="mt-1 space-y-1.5">
            {SKIPS.map((code) => (
              <li key={code}>
                <span className="text-ink">{reasonLabel(code)}</span>
                <span className="text-muted">
                  {" "}
                  — {t(`faq.skipWhy.${code}`)}
                </span>
              </li>
            ))}
          </ul>
          <Note>{t("faq.whyNotNote")}</Note>
        </Q>

        <Q q={t("faq.agentsQ")}>
          <p>{t("faq.agentsA1")}</p>
          <p>{t("faq.agentsA2")}</p>
          <p>{t("faq.agentsA3")}</p>
        </Q>

        <Q q={t("faq.stopQ")}>
          <p>
            <T k="faq.stopA1" />
          </p>
          <p>{t("faq.stopA2")}</p>
          <p>
            <T k="faq.stopA3" />
          </p>
        </Q>

        <Q q={t("faq.askQ")}>
          <p>{t("faq.askA1")}</p>
          <p>{t("faq.askA2")}</p>
        </Q>

        <Q q={t("faq.lessQ")}>
          <p>{t("faq.lessA1")}</p>
          <p>{t("faq.lessA2")}</p>
        </Q>

        <Q q={t("faq.stoppedQ")}>
          <p>{t("faq.stoppedA")}</p>
        </Q>

        <Q q={t("faq.filesQ")}>
          <p>{t("faq.filesA1")}</p>
          <p>
            <T k="faq.filesA2" />
          </p>
        </Q>

        <Q q={t("faq.apiQ")}>
          <p>
            <T k="faq.apiA1" />
          </p>
          <p>{t("faq.apiA2")}</p>
        </Q>
      </div>
    </div>
  );
}
