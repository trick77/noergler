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
  enabled: boolean;
  state: TeamState;
  /** -1 means at least one whole-project claim: unbounded, not a count. */
  repos: number;
  prs: number;
  last_run: string | null;
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
  breakdown: { outcome: string; reason?: string; label?: string; count: number }[];
}

export interface Team {
  slug: string;
  name: string;
  enabled: boolean;
  state: TeamState;
  repos: number;
  prs: number;
  last_run: string | null;
  claims: { project: string; repo?: string }[];
  /** COUNTS, not names: this route is unauthenticated and cross-team, and
   *  the author lists are Bitbucket usernames. */
  auto_review_authors: number;
  ignore_authors: number;
  exclude_repos: string[];
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
