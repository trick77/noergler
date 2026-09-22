// The developer-facing page: what somebody asks when noergler did something
// surprising to THEIR pull request. Everything here is behaviour the code
// already has; nothing on this page is configuration advice for operators,
// which lives in CONFIGURATION.md.
//
// Static prose, no endpoint: none of it is per-instance data. Where a value
// is a per-team default it says so, because teams.yaml can override almost
// all of it and a flat number here would be wrong for somebody.
import { Card, Note, column, h2, lede, page } from "./ui";

function Q({ q, children }: { q: string; children: React.ReactNode }) {
  return (
    <Card title={q}>
      <div className="space-y-2.5 text-[13.5px] leading-relaxed text-ink-dim">{children}</div>
    </Card>
  );
}

/** Code is a literal a reader might type or search for. */
function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded-ui-sm bg-active px-1 py-0.5 font-mono text-[12.5px] text-ink">
      {children}
    </code>
  );
}

// The eleven pre-flight exits, worded as review.SkipReason.Label() words
// them. Kept in that order: it is the order prepare reaches them.
const SKIPS: { label: string; why: string }[] = [
  { label: "AGENTS.md missing", why: "See below." },
  { label: "AGENTS.md over the token cap", why: "See below." },
  {
    label: "Author not in auto-review authors",
    why: "Ask your team admin to add you, or @mention the bot on the PR.",
  },
  { label: "Ignored author", why: "Usually a bot account. Ask your team admin if it is not." },
  { label: "Branch opt-out keyword", why: "Your branch name contains it. See below." },
  {
    label: "PR ignored (summary comment removed)",
    why: "Somebody deleted the summary comment. Nothing brings it back on this PR.",
  },
  { label: "PR cost cap reached", why: "@mention the bot to review anyway." },
  {
    label: "No reviewable files",
    why: "Only lockfiles, vendored code, generated output or binaries changed.",
  },
  { label: "Empty diff", why: "Bitbucket reports no changes." },
  { label: "HEAD unchanged since last review", why: "Push something." },
  { label: "Diff too large", why: "Only if your admin set a diff cap. Off by default." },
];

export function FaqPage() {
  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>FAQ</h2>
        <p className={lede}>
          Why noergler did or did not review your pull request. Defaults below; your team may have
          changed them.
        </p>

        <Q q="Why wasn't my PR reviewed?">
          <p>
            Open <strong>Runs</strong>, filter to Skipped or Failed, and find your PR. The reason is
            next to the outcome. What each one means:
          </p>
          <ul className="mt-1 space-y-1.5">
            {SKIPS.map((s) => (
              <li key={s.label}>
                <span className="text-ink">{s.label}</span>
                <span className="text-muted"> — {s.why}</span>
              </li>
            ))}
          </ul>
          <Note>Skipped means it never asked the model. Failed means it asked and something broke.</Note>
        </Q>

        <Q q="AGENTS.md">
          <p>
            It is what the review checks your code against: the conventions and constraints a new
            reviewer on your team would have to be told. Put it at the repo root.
          </p>
          <p>
            No AGENTS.md, no review. The summary comment says so rather than staying silent, so an
            unreviewed PR is never a mystery.
          </p>
          <p>
            Keep it short. The summary starts warning well before the size limit, and past the limit
            reviews stop entirely.
          </p>
        </Q>

        <Q q="How do I stop noergler reviewing a PR?">
          <p>
            Put <Code>noergloff</Code> anywhere in the source branch name:
            <Code>feature/noergloff-spike</Code> is never auto-reviewed.
          </p>
          <p>
            On a PR that is already open, delete noergler's summary comment instead. It stays away
            after that, and nothing brings it back on that PR.
          </p>
        </Q>

        <Q q="How do I ask noergler something?">
          <p>
            Mention the bot by its Bitbucket username in a PR comment. Ask a question and it
            answers; ask for a review and it runs one.
          </p>
          <p>
            Mentioning it overrides the auto-review list and the cost cap, so it works on PRs that
            would never be reviewed on their own.
          </p>
        </Q>

        <Q q="Why did the second review say less than the first?">
          <p>
            A new commit is reviewed against the last reviewed commit, not the whole PR, and
            findings already posted are not repeated.
          </p>
          <p>Reopening a declined PR starts over from scratch.</p>
        </Q>

        <Q q="Why did reviews stop halfway through my PR?">
          <p>
            The PR hit its cost cap; the running total is on the summary comment. Mention the bot to
            review anyway.
          </p>
        </Q>

        <Q q="Which files get reviewed?">
          <p>
            Source files that changed in the PR. Lockfiles, vendored directories, generated output
            and binaries are skipped.
          </p>
          <p>
            Your team can also exclude whole repos by name — <Code>*-infra</Code> by default. Those
            never get reviewed, even though the webhook still fires for them.
          </p>
        </Q>

        <Q q="Where is the API?">
          <p>
            <a href="/api/docs" className="text-ink underline hover:no-underline">
              /api/docs
            </a>
            . Claim and release repos, and change the author lists, without going through your
            noergler admin.
          </p>
          <p>
            You need your team's webhook secret, which the admin has. Claiming also needs your own
            Bitbucket token with project admin.
          </p>
        </Q>
      </div>
    </div>
  );
}
