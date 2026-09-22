// Formatters. Every number a person reads passes through one of these, so
// the page never decides locally how money or a duration looks.

/** money renders the API's decimal string, rounded to cents.
 *
 *  null is "unpriced", never $0.00: a run the gateway did not price is not a
 *  free run, and showing a zero invents a fact. A priced sub-cent run does
 *  read $0.00, which is what rounding to cents means; the two stay
 *  distinguishable because only one of them says "unpriced".
 *
 *  Rounding happens HERE, at render, and nowhere else: the wire keeps its
 *  exact decimal string, the DB stays BIGINT nano-USD and the riptide edge
 *  keeps every digit. The string is rounded AS a string rather than parsed,
 *  because a float cannot hold it: Number("1.005") is 1.00499..., so
 *  toFixed(2) answers $1.00. A 3-decimal string ending in 5 is the common
 *  case here, not a corner. */
export function money(usd: string | null): string {
  if (usd === null) return "unpriced";
  return `$${roundToCents(usd)}`;
}

/** roundToCents rounds a decimal string half-up, on the digits. A string that
 *  is not a plain decimal is returned unchanged rather than mangled: it came
 *  from the API, and inventing a number for it would be worse than showing
 *  what arrived. */
function roundToCents(usd: string): string {
  const m = /^(-?)(\d+)(?:\.(\d*))?$/.exec(usd.trim());
  if (m === null) return usd;
  const [, sign, whole, frac = ""] = m;

  // The third decimal decides. A digit beyond it cannot change a half-up
  // rounding the third has already settled, so the rest is not consulted.
  const cents = BigInt(whole) * 100n + BigInt((frac + "00").slice(0, 2));
  const rounded = frac.charCodeAt(2) >= 53 ? cents + 1n : cents;

  const digits = rounded.toString().padStart(3, "0");
  return `${sign}${digits.slice(0, -2)}.${digits.slice(-2)}`;
}

/** duration is a wall-clock span for a table cell. */
export function duration(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  // Round to whole seconds FIRST, then split. Rounding the remainder
  // independently of the minutes prints "1m 60s" for 119500ms: the minutes
  // floor to 1 while the 59.5s remainder rounds up to 60.
  const total = Math.round(s);
  return `${Math.floor(total / 60)}m ${total % 60}s`;
}

/** ago is a coarse relative time. Coarse on purpose: a live panel that
 *  re-renders a seconds counter every tick draws the eye to the clock rather
 *  than to what changed. */
export function ago(iso: string | null, now: number = Date.now()): string {
  if (iso === null) return "never";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const s = Math.max(0, Math.round((now - then) / 1000));
  if (s < 10) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/** elapsedSince is the live "running for" figure. */
export function elapsedSince(iso: string, now: number = Date.now()): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  return duration(Math.max(0, now - then));
}

/** tokens abbreviates a count. Thousands separators at this magnitude make a
 *  table column that is mostly punctuation. */
export function tokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}K`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

/** outcomeTone maps an outcome to its pill. Three tones, not one per
 *  outcome: ok, a decision the pipeline made, and something that went wrong.
 *  The pill always carries the word too, because the ok/skip/fail hues sit at
 *  dE 7.6 under protanopia and colour alone would not separate them. */
export type Tone = "ok" | "skip" | "fail";

export function outcomeTone(outcome: string): Tone {
  if (outcome === "ok") return "ok";
  if (outcome === "skipped") return "skip";
  return "fail";
}

/** teamTone maps a team's state to its pill.
 *
 *  no_repos is ochre, not green and not red: the team is configured and
 *  running, and it is also reviewing nothing. Green would claim it is
 *  working; red would claim it is broken. Ochre is "your move", which is
 *  exactly right, because somebody has to claim a repository. */
export function teamTone(state: string): Tone {
  if (state === "ready") return "ok";
  if (state === "no_repos") return "skip";
  return "fail";
}

/** teamStateLabel is the word on the pill. An unknown state prints itself
 *  rather than rendering blank. */
export function teamStateLabel(state: string): string {
  if (state === "ready") return "ready";
  if (state === "no_repos") return "no repos";
  if (state === "disabled") return "disabled";
  return state;
}

/** prUrl turns a PROJECT/repo#id tag into its Bitbucket URL, or null when it
 *  cannot: an empty base (BITBUCKET_URL unset) or a tag that does not have
 *  that shape. The caller renders plain text for a null rather than a broken
 *  href.
 *
 *  The tag is built by string formatting on the server, not by a parser, so
 *  this one must not assume it parses.
 *
 *  This is the BROWSER url, which is why it carries no /rest/api/1.0: the
 *  Go client's prPath builds the REST path for the same pull request and is
 *  deliberately a different shape. */
export function prUrl(tag: string, base: string): string | null {
  if (base === "") return null;
  const m = /^([^/]+)\/([^#]+)#(\d+)$/.exec(tag);
  if (m === null) return null;
  const [, project, repo, id] = m;
  return `${base}/projects/${encodeURIComponent(project)}/repos/${encodeURIComponent(repo)}/pull-requests/${id}`;
}

/** byName sorts teams the way the page reads them: by display name, folding
 *  case and accents, so "Diecibärg" files under D rather than after Z.
 *
 *  A COPY, never in place: the array comes from the poller and is reused
 *  between renders, so sorting it where it lies mutates state React is
 *  holding. The server's own order is by slug, which looks arbitrary once
 *  the page stops printing slugs. */
export function byName<T extends { name: string }>(teams: T[]): T[] {
  return [...teams].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
}

/** scope renders a team's repository count, keeping "every repo in the
 *  project" distinct from any particular number. */
export function scope(repos: number): string {
  if (repos < 0) return "whole project";
  if (repos === 0) return "no repos";
  return `${repos} repo${repos === 1 ? "" : "s"}`;
}
