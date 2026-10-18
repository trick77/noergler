// English UI catalog, the source of truth for the shape: de.ts must mirror it
// (`satisfies Catalog`), so a key added here and forgotten there fails the
// build. Grouped by page. Review text (findings, comments, replies) is never
// in here: it keeps the language it was produced in.
//
// Outcome words are reader words in both languages ("timed out", not the
// stored code timed_out). A failure and a skip must never share a word:
// the word is the cue that does not depend on colour.
//
// The `reason` block repeats review.SkipReason.Label() word for word, and
// TestSkipReasonLabelsMatchTheUICatalog fails when the two drift.
export const en = {
  nav: {
    label: "Pages",
    live: "Live",
    runs: "Runs",
    metrics: "Metrics",
    teams: "Teams",
    faq: "FAQ",
    api: "API",
  },
  common: {
    failed:
      "Could not load {{what}}. The instance may be starting, or the database may be unreachable.",
    unpriced: "unpriced",
    never: "never",
    justNow: "just now",
    secondsAgo: "{{count}}s ago",
    minutesAgo: "{{count}}m ago",
    hoursAgo: "{{count}}h ago",
    daysAgo_one: "{{count}}d ago",
    daysAgo_other: "{{count}}d ago",
  },
  outcome: {
    ok: "ok",
    skipped: "skipped",
    timed_out: "timed out",
    unparseable: "unparseable",
    too_large: "prompt too large",
    error: "error",
  },
  // The pill word per review.SkipReason. Short: the long label beside a pill
  // wrapped the Outcome cell into three or four lines.
  skipWord: {
    not_auto_review_author: "not opted in",
    ignored_author: "ignored",
    pr_ignored: "PR ignored",
    branch_opt_out: "opted out",
    agents_md_missing: "no AGENTS.md",
    agents_md_too_large: "AGENTS.md too big",
    pr_cost_cap: "cost cap",
    no_reviewable_files: "nothing to review",
    empty_diff: "empty diff",
    head_unchanged: "unchanged",
    diff_too_large: "diff too large",
    repo_not_claimed: "unclaimed",
  },
  // The long form, worded as review.SkipReason.Label() words it. A code this
  // build does not know falls back to the label the server sent.
  reason: {
    not_auto_review_author: "Author not in auto-review authors",
    ignored_author: "Ignored author",
    pr_ignored: "PR ignored (summary comment removed)",
    branch_opt_out: "Branch opt-out keyword",
    agents_md_missing: "AGENTS.md missing",
    agents_md_too_large: "AGENTS.md over the token cap",
    pr_cost_cap: "PR cost cap reached",
    no_reviewable_files: "No reviewable files",
    empty_diff: "Empty diff",
    head_unchanged: "HEAD unchanged since last review",
    diff_too_large: "Diff too large",
    repo_not_claimed: "Repo no longer claimed by the team",
  },
  teamState: {
    ready: "ready",
    no_repos: "no repos",
    disabled: "disabled",
  },
  live: {
    lede: "What is running right now.",
    ledeLong:
      "What is running right now. Several reviews can be waiting on the model at once, but only one at a time talks to Bitbucket, so a busy queue drains in order.",
    failedWhat: "the live panel",
    refreshFailing: "Refresh failing. Showing the reading from {{ago}}.",
    pool: "Inference pool",
    slotsBusy: "{{busy}} of {{capacity}} slots busy",
    maxPerTeam: "max {{count}} per team",
    nothingRunning: "Nothing running.",
    queue: "Queue",
    waitingCount: "{{count}} waiting",
    inFlightExcluded: "in-flight excluded",
    queueEmpty: "Queue is empty.",
    pullRequest: "Pull request",
    team: "Team",
    waiting: "Waiting",
    teams: "Teams",
    noRuns: "No runs yet.",
    lastOutcome: "Last outcome",
    prs: "PRs",
    lastReviewed: "Last reviewed",
    lastRun: "Last run",
    teamsNote:
      "Last run counts every outcome, skipped and failed included. Last reviewed is the last one that succeeded.",
  },
  runs: {
    lede: "What noergler did with every PR this month, including the ones it decided not to review, and the latest 50 runs below. Skipped means it never asked the model. Failed means it asked and something went wrong.",
    failedWhat: "the runs feed",
    tileRuns: "Runs",
    tileReviewed: "Reviewed",
    tileSkipped: "Skipped",
    tileFailed: "Failed",
    latest: "Latest 50 runs",
    filterAll: "All",
    filterFailed: "Failed",
    filterSkipped: "Skipped",
    empty: "No runs recorded yet.",
    emptyFailed: "No failed runs recorded yet.",
    emptySkipped: "No skipped runs recorded yet.",
    pullRequest: "Pull request",
    team: "Team",
    outcome: "Outcome",
    findings: "Findings",
    elapsed: "Elapsed",
    cost: "Cost",
    when: "When",
    whySkipped: "Why runs were skipped",
    countsUnavailable: "Counts unavailable.",
    nothingSkipped: "Nothing was skipped.",
    reason: "Reason",
    count: "Count",
  },
  metrics: {
    failedWhat: "metrics",
    lastDays_one: "the last day",
    lastDays_other: "the last {{count}} days",
    thisMonth: "this month",
    lede: "What this month cost and how much got reviewed. Runs the AI Gateway never priced are counted on their own rather than added in as zero, so the total is not quietly too low.",
    cost: "Cost",
    reviewed: "Reviewed",
    findings: "Findings",
    unpriced: "Unpriced",
    runsNote: "runs",
    costTrend: "Cost trend",
    reviewedTrend: "Reviewed trend",
    costPerDay: "Cost per day",
    spend: "Spend",
    spendNote: "USD, priced runs only",
    noPriced: "No priced runs in this window.",
    costChart: "Cost per day for {{title}}, by team",
    more: "{{count}} more",
    runsPerDay: "Runs per day",
    throughput: "Throughput",
    throughputNote: "reviewed, skipped and failed runs",
    noRuns: "No runs in this window.",
    runsChart: "Runs per day for {{title}}",
    seriesReviewed: "reviewed",
    seriesSkipped: "skipped",
    seriesFailed: "failed",
    byTeam: "By team",
    nothingReviewed: "Nothing reviewed in this window.",
    team: "Team",
    tokensIn: "Tokens in",
    tokensOut: "Tokens out",
    byTeamNote:
      "Ordered by spend. Costs the AI Gateway never reported are missing from this table.",
  },
  teams: {
    failedWhat: "the team roster",
    lede: "Read-only. A team is a webhook path, its own HMAC secret and the projects and repositories it has claimed.",
    prs_one: "{{count}} PR",
    prs_other: "{{count}} PRs",
    lastReviewed: "last reviewed {{ago}}",
    lastRun: "last run {{ago}}",
    noClaims:
      "No projects and no repositories claimed, so this team reviews nothing.",
    claim: "Claim",
    scope: "Scope",
    scopeRepo: "repo",
    scopeProject: "project",
    excludedRepos: "Excluded repos:",
    autoReviewAuthors: "Auto-review authors:",
    ignoredAuthors: "Ignored authors:",
    disabledNote:
      "Look in the pod log for the <code>team_disabled</code> line to see why. It is not shown here because the reason can name an environment variable.",
    none: "No teams configured.",
  },
  faq: {
    lede: "Why noergler did or did not review your pull request. Defaults below; your team may have changed them.",
    // Per review.SkipReason, in the order prepare reaches them.
    skipWhy: {
      agents_md_missing: "See below.",
      agents_md_too_large: "See below.",
      not_auto_review_author:
        "Ask your team admin to add you, or @mention the bot on the PR.",
      ignored_author:
        "Usually a bot account. Ask your team admin if it is not.",
      branch_opt_out: "Your branch name contains it. See below.",
      pr_ignored:
        "Somebody deleted the summary comment. Nothing brings it back on this PR.",
      pr_cost_cap: "@mention the bot to review anyway.",
      no_reviewable_files:
        "Only lockfiles, vendored code, generated output or binaries changed.",
      empty_diff: "Bitbucket reports no changes.",
      head_unchanged: "Push something.",
      diff_too_large: "Only if your admin set a diff cap. Off by default.",
    },
    whyNotQ: "Why wasn't my PR reviewed?",
    whyNotA:
      "Open <strong>Runs</strong>, filter to Skipped or Failed, and find your PR. The reason is next to the outcome. What each one means:",
    whyNotNote:
      "Skipped means it never asked the model. Failed means it asked and something broke.",
    agentsQ: "AGENTS.md",
    agentsA1:
      "It is what the review checks your code against: the conventions and constraints a new reviewer on your team would have to be told. Put it at the repo root.",
    agentsA2:
      "No AGENTS.md, no review. The summary comment says so rather than staying silent, so an unreviewed PR is never a mystery.",
    agentsA3:
      "Keep it short. The summary starts warning well before the size limit, and past the limit reviews stop entirely.",
    stopQ: "How do I stop noergler reviewing a PR?",
    stopA1:
      "Put <code>noergloff</code> anywhere in the source branch name: <code>feature/noergloff-spike</code> is never auto-reviewed.",
    stopA2:
      "On a PR that is already open, delete noergler's summary comment instead. It stays away after that, and nothing brings it back on that PR.",
    askQ: "How do I ask noergler something?",
    askA1:
      "Mention the bot by its Bitbucket username in a PR comment. Ask a question and it answers; ask for a review and it runs one.",
    askA2:
      "Mentioning it overrides the auto-review list and the cost cap, so it works on PRs that would never be reviewed on their own.",
    lessQ: "Why did the second review say less than the first?",
    lessA1:
      "A new commit is reviewed against the last reviewed commit, not the whole PR, and findings already posted are not repeated.",
    lessA2: "Reopening a declined PR starts over from scratch.",
    stoppedQ: "Why did reviews stop halfway through my PR?",
    stoppedA:
      "The PR hit its cost cap; the running total is on the summary comment. Mention the bot to review anyway.",
    filesQ: "Which files get reviewed?",
    filesA1:
      "Source files that changed in the PR. Lockfiles, vendored directories, generated output and binaries are skipped.",
    filesA2:
      "Your team can also exclude whole repos by name — <code>*-infra</code> by default. Those never get reviewed, even though the webhook still fires for them.",
    apiQ: "Where is the API?",
    apiA1:
      "<docs>/api/docs</docs>. Claim and release repos, and change the author lists, without going through your noergler admin.",
    apiA2:
      "You need your team's webhook secret, which the admin has. Claiming also needs your own Bitbucket token with project admin.",
  },
};

export type Catalog = typeof en;
