// Formatters. Every number a person reads passes through one of these, so
// the page never decides locally how money or a duration looks.

/** money renders the API's decimal string. null is "unpriced", never $0.000:
 *  a run the gateway did not price is not a free run, and showing a zero
 *  invents a fact. The string is passed through rather than parsed, because
 *  parsing it to a float would round the money the backend took care to keep
 *  exact. */
export function money(usd: string | null): string {
  if (usd === null) return "unpriced";
  return `$${usd}`;
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

/** scope renders a team's repository count, keeping "every repo in the
 *  project" distinct from any particular number. */
export function scope(repos: number): string {
  if (repos < 0) return "whole project";
  if (repos === 0) return "no repos";
  return `${repos} repo${repos === 1 ? "" : "s"}`;
}
