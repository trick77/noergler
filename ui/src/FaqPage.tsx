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
  {
    label: "AGENTS.md missing",
    why: "The repository has no AGENTS.md. noergler reviews against the conventions that file states, so without one there is nothing to review against.",
  },
  {
    label: "AGENTS.md over the token cap",
    why: "The file is too large to carry in the prompt. There is a warning threshold below the cap, so a summary usually says so before reviews stop.",
  },
  {
    label: "Author not in auto-review authors",
    why: "The team restricts automatic reviews to a list of authors and yours is not on it. An @mention still works.",
  },
  {
    label: "Ignored author",
    why: "The author is on the team's ignore list, which is normally where CI and dependency bots go.",
  },
  {
    label: "Branch opt-out keyword",
    why: "The source branch name contains the opt-out keyword. See below.",
  },
  {
    label: "PR ignored (summary comment removed)",
    why: "Somebody deleted noergler's summary comment on this PR, which is how you tell it to leave the PR alone.",
  },
  {
    label: "PR cost cap reached",
    why: "This PR has already spent its budget. See below.",
  },
  {
    label: "No reviewable files",
    why: "Everything that changed is a file type noergler skips: lockfiles, vendored code, generated output, binaries.",
  },
  { label: "Empty diff", why: "Nothing changed, as far as Bitbucket reports." },
  {
    label: "HEAD unchanged since last review",
    why: "The commit was already reviewed. Pushing the same HEAD again does not re-run it.",
  },
  {
    label: "Diff too large",
    why: "Only when the operator set a diff cap. It is off by default.",
  },
];

export function FaqPage() {
  return (
    <div className={page}>
      <div className={column}>
        <h2 className={h2}>FAQ</h2>
        <p className={lede}>
          Why noergler did, or did not, review your pull request. Most of the numbers below are
          per-team defaults; your team may have changed them.
        </p>

        <Q q="Why wasn't my PR reviewed?">
          <p>
            Every attempt is recorded, including the ones that produced no review, and the reason is
            on the row. Open <strong>Runs</strong> and filter to Skipped or Failed: the reason is
            beside the outcome. These are the reasons a review is skipped before it starts:
          </p>
          <ul className="mt-1 space-y-1.5">
            {SKIPS.map((s) => (
              <li key={s.label}>
                <span className="text-ink">{s.label}</span>
                <span className="text-muted"> — {s.why}</span>
              </li>
            ))}
          </ul>
          <Note>
            A failure is different from a skip: a skip is a decision noergler made before asking the
            model, a failure is the gateway or the parser going wrong afterwards.
          </Note>
        </Q>

        <Q q="How do I stop noergler reviewing a PR?">
          <p>
            Put the opt-out keyword in the <strong>source branch name</strong>. By default that
            keyword is <Code>noergloff</Code>, so <Code>feature/noergloff-spike</Code> is never
            auto-reviewed. It is a substring match on the branch name, and your team can change the
            word.
          </p>
          <p>
            For a PR that is already open, delete noergler's summary comment. It takes that as "stop
            touching this one" and will not come back to it.
          </p>
        </Q>

        <Q q="How do I ask noergler something?">
          <p>
            Mention the bot by its Bitbucket username in a PR comment. A question gets an answer; a
            comment asking for a review runs a full one, even when the PR would not have been
            auto-reviewed — the person asking is the authorization.
          </p>
          <p>
            An @mention also works after the PR hit its cost cap, and for an author who is not on
            the auto-review list.
          </p>
        </Q>

        <Q q="Why does my repo need an AGENTS.md?">
          <p>
            It is what noergler reviews against: the conventions, constraints and decisions that a
            reviewer would otherwise have to already know. Without one, a review has nothing to hold
            the code to, so by default the review is skipped and the summary says so.
          </p>
          <p>
            Size matters too. Past a warning threshold the summary starts warning; past the maximum
            the review stops. Both are token counts and both are per-team.
          </p>
        </Q>

        <Q q="Why did the second review say less than the first?">
          <p>
            Re-reviews are incremental: pushing a new commit reviews what changed since the last
            reviewed commit, not the whole PR again. Findings already posted are not repeated.
          </p>
          <p>
            Pushing nothing new changes nothing — an unchanged HEAD is skipped. A declined PR that is
            reopened starts fresh.
          </p>
        </Q>

        <Q q="Why did reviews stop halfway through my PR?">
          <p>
            Each PR has a cost cap. Once the reviews on it have spent that much, automatic reviews
            stop, and the per-PR total is on the summary. An @mention still runs a review past the
            cap.
          </p>
        </Q>

        <Q q="Which files get reviewed?">
          <p>
            Source files that changed in the PR. Lockfiles, vendored directories, generated output
            and binaries are skipped: a diff full of them is mostly noise, and they are not what a
            reviewer reads.
          </p>
          <p>
            Whole repositories can be excluded too. A team claims a project, and its exclude list
            then drops repositories inside it — <Code>*-infra</Code> by default — so a repo can be
            covered by the webhook and still never be reviewed.
          </p>
        </Q>

        <Q q="Where is the API?">
          <p>
            The team self-service API — claiming projects and repositories, and the review settings
            (auto-review authors, ignored authors, excluded repositories) — is documented at{" "}
            <a href="/api/docs" className="text-ink underline hover:no-underline">
              /api/docs
            </a>
            .
          </p>
          <p>
            It authenticates with your team's webhook secret, which your noergler admin has.
            Claiming or releasing repositories additionally needs your own Bitbucket token, with
            project admin on the target.
          </p>
        </Q>
      </div>
    </div>
  );
}
