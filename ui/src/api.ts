// The dashboard's wire types. They mirror the structs in
// backend/internal/api/dashboard.go; a field that is nullable there is
// `| null` here, never optional, so a missing value has to be handled rather
// than silently read as undefined.

export interface LiveItem {
  tag: string;
  /** The slug: what the logs and the webhook path use, and what the page
   *  keys off. team_name is what it prints. */
  team: string;
  team_name: string;
  kind?: string;
  since: string;
  waited_ms?: number;
}

/** ready | no_repos | disabled. A team can be enabled and still own no
 *  repositories, which is neither of the other two. */
export type TeamState = "ready" | "no_repos" | "disabled";

export interface LiveTeam {
  slug: string;
  /** Display name from teams.yaml, defaulted to the slug by the server. */
  name: string;
  prs: number;
  /** The last success. */
  last_reviewed: string | null;
  /** The last run of any outcome. Never null: live lists only teams that
   *  have run. */
  last_run: string;
  last_outcome: string;
  last_reason?: string;
  last_reason_label?: string;
}

export interface Live {
  pool_capacity: number;
  pool_per_team: number;
  staged: number;
  depth: number;
  running: LiveItem[];
  waiting: LiveItem[];
  teams: LiveTeam[];
  /** The instance's Bitbucket base, for linking a PR tag. Empty when
   *  BITBUCKET_URL is unset; the page then shows the tag as plain text. */
  bitbucket_url: string;
}

export interface Run {
  /** The attempt id, the key of its run detail page. */
  id: number;
  tag: string;
  team: string;
  team_name: string;
  kind: string;
  outcome: string;
  reason?: string;
  reason_label?: string;
  elapsed_ms: number | null;
  findings: number | null;
  // A decimal STRING, or null for an unpriced run. Never a number: an
  // unpriced run is not a free one, and a float would round the money.
  cost_usd: string | null;
  created_at: string;
}

export interface Runs {
  runs: Run[];
  /** As on Live: this page polls its own endpoint, so it cannot read the
   *  Bitbucket base off the live response. */
  bitbucket_url: string;
}

export interface Totals {
  runs: number;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  findings_posted: number;
  cost_usd: string | null;
  unpriced_runs: number;
}

export interface Metrics {
  since: string;
  /** Exclusive end of the window. */
  until: string;
  /** "month" for the current calendar month, "rolling" for a ?days= window. */
  window: string;
  totals: Totals;
  by_team: (Totals & { team: string; team_name: string })[];
  daily: { day: string; team: string; runs: number; cost_usd: string | null }[];
  daily_attempts: { day: string; outcome: string; count: number }[];
  breakdown: {
    outcome: string;
    reason?: string;
    label?: string;
    count: number;
  }[];
}

export interface Team {
  slug: string;
  name: string;
  enabled: boolean;
  state: TeamState;
  repos: number;
  prs: number;
  /** Last success, and last run of any outcome. */
  last_reviewed: string | null;
  last_run: string | null;
  claims: { project: string; repo?: string }[];
  /** COUNTS, not names: this route is unauthenticated and cross-team, and
   *  the author lists are Bitbucket usernames. */
  auto_review_authors: number;
  ignore_authors: number;
  exclude_repos: string[];
  /** Branch globs: a matching source or target branch is not auto-reviewed. */
  exclude_source_branches: string[];
  exclude_target_branches: string[];
  /** `<profile>-<effort>`, the label every run records. Absent for a
   *  disabled team: it has no started client. */
  model?: string;
  /** "team" when teams.yaml gives the team a model or effort other than the
   *  instance's. Absent with model. */
  model_source?: "team" | "default";
  /** Forwarding as of the last call (startup ping, then each rollup), never
   *  the URL. Absent for a disabled team. */
  riptide?: ForwardStatus;
  /** The Almanach note feed, the same fixed words as riptide: off without
   *  an almanach block, else as of the last call. Absent for a disabled
   *  team. */
  almanach?: ForwardStatus;
  /** The Bitbucket and Jira accounts the team works with: the shared ones,
   *  or its own (pushed through the team API). A fixed status, never a
   *  reason. Null for a disabled team. */
  bitbucket_account: BitbucketAccount | null;
  jira_account: JiraAccount | null;
}

/** ForwardStatus is a forwarder's state (riptide, Almanach), the server's
 *  fixed words as of its last call: never a reason. */
export type ForwardStatus = "off" | "ok" | "failing" | "rejected";

/** AccountStatus is an own account's health; null on a shared one. */
export type AccountStatus =
  "ok" | "rejected" | "unreachable" | "undecryptable" | "username_mismatch";

export interface JiraAccount {
  source: "team" | "shared";
  status: AccountStatus | null;
  since: string | null;
}

export interface BitbucketAccount extends JiraAccount {
  /** The bot the team @mentions: its own, or the shared one. */
  username: string;
}

/** RunFacts is the review_runs row behind a successful attempt. */
export interface RunFacts {
  incremental: boolean;
  from_commit?: string;
  to_commit?: string;
  model: string;
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  findings_posted: number;
  lines_added: number;
  lines_removed: number;
  files_changed: number;
}

/** RunDetail is GET prs/{token}/runs/{n}: one attempt of the PR and what was
 *  recorded about it. Full finding text (secret-like text withheld); money
 *  as decimal strings, null for unpriced or "the gateway did not send it",
 *  never a zero. */
export interface RunDetail {
  /** The run's number on its PR, 1 for the oldest. */
  n: number;
  tag: string;
  title: string;
  team: string;
  team_name: string;
  kind: string;
  outcome: string;
  reason?: string;
  reason_label?: string;
  elapsed_ms: number | null;
  created_at: string;
  cost_usd: string | null;
  run: RunFacts | null;
  call: RunCall | null;
  findings: RunFinding[];
  fixed: RunFinding[];
  bitbucket_url: string;
  /** The review's pipeline record; null for a run from before it existed,
   *  or a skip. */
  detail: RunPipeline | null;
}

/** PRPage is GET prs/{token}: the PR, every run of it, every finding posted
 *  on it, the files its reviews saw and what it cost. */
export interface PRPage {
  tag: string;
  title: string;
  team: string;
  team_name: string;
  /** open | merged | declined */
  state: string;
  /** The cap's figure, never a sum of the rows. */
  cost_usd: string | null;
  cap_usd: string | null;
  /** The whole PR's reviewable diff; null when no run read it. */
  scope: { added: number; removed: number; files: number } | null;
  /** Tokens and wall clock over every run, the unlisted ones included. */
  prompt_tokens: number;
  cached_tokens: number;
  completion_tokens: number;
  elapsed_ms: number;
  /** The waits in the recorded stages of the runs listed. */
  wait_ms: number;
  /** Every run of the PR; runs lists the newest of them. */
  runs_total: number;
  /** Oldest first, numbered over the whole history. */
  runs: PRRun[];
  /** Older runs are not listed. */
  more_runs: boolean;
  findings: PRFinding[];
  files: PRFile[];
  /** The PR's reply and Q&A calls, newest first. They count toward cost_usd
   *  and the cap. */
  calls: CommentCall[];
  more_calls: boolean;
  /** The list could not be read: empty means unknown, not none. */
  calls_failed: boolean;
  bitbucket_url: string;
}

export interface PRRun {
  n: number;
  kind: string;
  outcome: string;
  reason?: string;
  reason_label?: string;
  elapsed_ms: number | null;
  created_at: string;
  cost_usd: string | null;
  run: RunFacts | null;
  /** Earlier findings this run retired. */
  fixed: number;
  /** The recorded stages in three parts; null without a record. */
  time: { work_ms: number; wait_ms: number; gateway_ms: number } | null;
}

/** PRFinding is a posted finding: headline only, the runs by number (null
 *  when that run is not in the history listed). */
export interface PRFinding {
  severity: string;
  file: string;
  line: number;
  headline: string;
  comment_id: number | null;
  replied: boolean;
  fixed: boolean;
  posted_in: number | null;
  fixed_in: number | null;
}

/** PRFile is a file any run's record lists, with how each run sent it. */
export interface PRFile {
  path: string;
  runs: { n: number; sent: string }[];
  findings: number;
}

/** RunPipeline is the review's own record of the run. */
export interface RunPipeline {
  /** The reviewed commit, for links into the code. */
  commit?: string;
  stages: { name: string; ms: number }[];
  prompt: {
    template: number;
    files: number;
    cumulative_diff: number;
    previously_posted: number;
    repo_instructions: number;
    ticket: number;
    pr_description: number;
    total: number;
    budget: number;
  };
  files: {
    path: string;
    sent: string;
    added: number;
    removed: number;
    tokens: number;
  }[];
  cross_file: string[];
  agents_md: { found: boolean; tokens: number; warn_tokens: number };
  ticket?: string;
  parent_ticket?: string;
  previously_posted: { shown: number; total: number };
  decisions: {
    file: string;
    line: number;
    severity: string;
    headline: string;
    comment?: string;
    suggestion?: string;
    evidence?: string[];
    decision: string;
    reason?: string;
  }[];
  moved: { file: string; from: number; to: number }[];
  verdict?: { model: string; posted: string };
  /** Bugs in code the PR did not change: never posted, shown only here. */
  pre_existing: {
    file: string;
    line: number;
    headline: string;
    explanation: string;
    evidence?: string[];
  }[];
  /** Requests per upstream, by name. */
  http: { upstream: string; calls: number }[];
}

/** RunCall is what the AI Gateway reported about the call. Lanes at six
 *  decimals; output_tokens includes reasoning_tokens. */
export interface RunCall {
  profile: string;
  model?: string;
  deployment?: string;
  reasoning_sent?: string;
  finish_reason?: string;
  call_id?: string;
  litellm_version?: string;
  input_tokens: number | null;
  input_uncached_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  output_tokens: number | null;
  reasoning_tokens: number | null;
  cost_usd: string | null;
  cost_input_usd: string | null;
  cost_cache_read_usd: string | null;
  cost_cache_write_usd: string | null;
  cost_output_usd: string | null;
  cost_reasoning_usd: string | null;
  cost_tool_usage_usd: string | null;
  key_spend_usd: string | null;
  headers_ms: number | null;
  total_ms: number | null;
  warnings: string[];
}

export interface RunFinding {
  severity: string;
  confidence: number | null;
  file: string;
  line: number;
  headline: string;
  /** The comment as posted, the suggestion and the author's dispute. */
  comment: string;
  suggestion: string | null;
  dispute: string | null;
  comment_id: number | null;
  replied: boolean;
  fixed: boolean;
}

export interface CommentCall {
  /** reply | question */
  kind: string;
  comment_id: number | null;
  model: string;
  cost_usd: string | null;
  created_at: string;
}

/** get fetches one dashboard endpoint. Relative, so vite proxies it in dev
 *  and it is same-origin in production. */
export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`/api/dashboard/${path}`, { signal });
  if (!res.ok) {
    throw new Error(`${path}: HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}
