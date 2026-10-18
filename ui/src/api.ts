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
  /** `<profile>-<effort>`, the label every run records. Absent for a
   *  disabled team: it has no started client. */
  model?: string;
}

/** RunDetail is GET runs/{id}: one attempt and what was recorded about it.
 *  Headlines only, never a finding's text; money as decimal strings, null
 *  for unpriced or "the gateway did not send it", never a zero. */
export interface RunDetail {
  id: number;
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
  run: {
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
  } | null;
  call: RunCall | null;
  findings: RunFinding[];
  fixed: RunFinding[];
  pr: {
    attempts: PRAttempt[];
    /** The history around this run is cut on that side. */
    more_before: boolean;
    more_after: boolean;
    cost_usd: string | null;
    cap_usd: string | null;
  };
  bitbucket_url: string;
  /** The review's pipeline record; null for a run from before it existed,
   *  or a skip. */
  detail: RunPipeline | null;
}

/** RunPipeline is the review's own record of the run: headlines only. */
export interface RunPipeline {
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
    decision: string;
    reason?: string;
  }[];
  moved: { file: string; from: number; to: number }[];
  verdict?: { model: string; posted: string };
  http: Record<string, number>;
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
  comment_id: number | null;
  replied: boolean;
  fixed: boolean;
  /** The page of the run that posted it; null when that attempt is gone. */
  from_attempt_id: number | null;
}

export interface PRAttempt {
  id: number;
  kind: string;
  outcome: string;
  reason?: string;
  reason_label?: string;
  elapsed_ms: number | null;
  created_at: string;
  findings: number | null;
  cost_usd: string | null;
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
