// Formatters. Every number a person reads passes through one of these, so
// the page never decides locally how money or a duration looks. Their words
// come from the i18n catalog; the shapes of numbers do not change with it.
import i18n from "./i18n";
import { en, type Catalog } from "./i18n/en";

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
  if (usd === null) return i18n.t("common.unpriced");
  return `$${roundToCents(usd)}`;
}

/** moneyTick formats a cost axis with the fewest decimals (at least two, at
 *  most six) that print every tick exactly: a fixed toFixed(2) printed every
 *  tick of a sub-cent axis as $0.00, and a step of 0.0075 needs four.
 *  Ticks are axis positions, not amounts, so a float is fine. */
export function moneyTick(ticks: number[]): (v: number) => string {
  let decimals = 2;
  const exact = (v: number) =>
    Math.abs(Number(v.toFixed(decimals)) - v) < 1e-12;
  while (decimals < 6 && !ticks.every(exact)) decimals++;
  return (v) => `$${v.toFixed(decimals)}`;
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
  if (iso === null) return i18n.t("common.never");
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const s = Math.max(0, Math.round((now - then) / 1000));
  if (s < 10) return i18n.t("common.justNow");
  if (s < 60) return i18n.t("common.secondsAgo", { count: s });
  const m = Math.floor(s / 60);
  if (m < 60) return i18n.t("common.minutesAgo", { count: m });
  const h = Math.floor(m / 60);
  if (h < 24) return i18n.t("common.hoursAgo", { count: h });
  return i18n.t("common.daysAgo", { count: Math.floor(h / 24) });
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

/** outcomeTone maps an outcome to its pill. Four tones, not one per
 *  outcome: ok, a decision the pipeline made, something that went wrong, and
 *  none for no value at all (a team that never ran), grey because it is none
 *  of the three. Only outcomeTone uses none: teamTone keeps an empty state
 *  red on purpose, since the server always names one. The pill always
 *  carries the word too, because the
 *  ok/skip/fail hues sit at dE 7.6 under protanopia and colour alone would
 *  not separate them. */
export type Tone = "ok" | "skip" | "fail" | "none";

export function outcomeTone(outcome: string): Tone {
  if (outcome === "ok") return "ok";
  if (outcome === "") return "none";
  if (outcome === "skipped") return "skip";
  return "fail";
}

/** outcomeWord is the word on a run's outcome pill. A skip names its reason;
 *  a skip whose reason this build does not know still says "skipped", so a
 *  row written by a newer binary never renders blank. Every skip keeps the
 *  skip tone, so the Skipped filter and counts still read as one group. */
export function outcomeWord(outcome: string, reason?: string): string {
  if (outcome !== "skipped") {
    return known("outcome", outcome)
      ? i18n.t(`outcome.${outcome}`)
      : readable(outcome);
  }
  return reason && known("skipWord", reason)
    ? i18n.t(`skipWord.${reason}`)
    : i18n.t("outcome.skipped");
}

/** readable is the one fallback for a code this build has no word for (a
 *  newer binary wrote it): the code with spaces for underscores, and a dash
 *  for no code at all, so a pill is never blank. */
function readable(code: string): string {
  return code === "" ? "—" : code.replaceAll("_", " ");
}

/** outcomeTitle is the pill's tooltip: a skip's long reason, and for a
 *  failure the stored code (too_large), which is what the log and the
 *  database say. The pill shows a reader's word; the code an operator greps
 *  for stays one hover away, marked as a code so a German reader does not
 *  take it for a missed translation. */
export function outcomeTitle(
  outcome: string,
  reason?: string,
  serverLabel?: string,
): string | undefined {
  if (outcome === "ok" || outcome === "") return undefined;
  if (outcome === "skipped") return reasonLabel(reason, serverLabel);
  return codeText(outcome);
}

/** codeText marks a stored code as one ("Code: too_large") wherever the UI
 *  shows a code: on a failure's tooltip next to its word, or in place of a
 *  word this build lacks. The thing to grep for, never mistaken for a missed
 *  translation. */
function codeText(code: string): string {
  return i18n.t("common.code", { code });
}

/** known narrows a stored code to a key of its catalog group, so the lookup
 *  below it stays type-checked: a renamed group or a misspelt prefix fails
 *  the build instead of every pill quietly falling back. A code the catalog
 *  does not know (a newer binary wrote it) answers false and takes the
 *  caller's fallback. Asked of en: de mirrors its keys exactly (satisfies
 *  Catalog), so the answer holds for either language.
 *
 *  Only the plain code -> word groups qualify. A nested group (faq) or one
 *  with {{placeholders}} (common) would let a caller render an object or a
 *  raw template. */
type CodeGroup =
  | "outcome"
  | "skipWord"
  | "reason"
  | "teamState"
  | "severity"
  | "stage"
  | "sent"
  | "decision"
  | "dropReason";

/** known tells a code the catalog has from one a newer binary wrote, which
 *  the page prints as it came. */
export function known<G extends CodeGroup>(
  group: G,
  code: string,
): code is Extract<keyof Catalog[G], string> {
  return Object.hasOwn(en[group], code);
}

/** reasonLabel is the long form of a skip reason, for a tooltip or a
 *  table. Keys are review.SkipReason values, stored and stable; a code this
 *  build does not know falls back to the label the server sent, then to the
 *  code, marked as one (codeText): what an operator greps for. */
export function reasonLabel(
  reason?: string,
  serverLabel?: string,
): string | undefined {
  if (!reason) return serverLabel || undefined;
  return known("reason", reason)
    ? i18n.t(`reason.${reason}`)
    : serverLabel || codeText(reason);
}

/** severityWord is a finding's severity as the reader's word. The stored
 *  value is the model's own, so an unexpected one prints readably. */
export function severityWord(severity: string): string {
  return known("severity", severity)
    ? i18n.t(`severity.${severity}`)
    : readable(severity);
}

/** severityTone: red for an issue, grey for a suggestion (the model's two
 *  severities, review.txt) and for anything else. The word on the pill keeps
 *  it from being colour alone. */
export function severityTone(severity: string): Tone {
  return severity === "issue" ? "fail" : "none";
}

/** teamTone maps a team's state to its pill.
 *
 *  no_repos is ochre, not green and not red: the team is configured and
 *  running, and it is also reviewing nothing. Green would claim it is
 *  working; red would claim it is broken. Ochre is "your move", which is
 *  exactly right, because somebody has to claim a repository. */
export function teamTone(state: string): Tone {
  // No "" case: the server always names the state, so an empty one is a
  // broken contract and stays red like any unknown state. (An empty outcome
  // is different: a team that never ran really has none.)
  if (state === "ready") return "ok";
  if (state === "no_repos") return "skip";
  return "fail";
}

/** teamStateLabel is the word on the pill. An unknown state prints readably
 *  (readable) rather than rendering blank, and teamStateTitle keeps its
 *  stored code on the tooltip. */
export function teamStateLabel(state: string): string {
  return known("teamState", state)
    ? i18n.t(`teamState.${state}`)
    : readable(state);
}

/** teamStateTitle is the code of a state this build has no word for, the
 *  thing to grep for; a known state needs no tooltip. */
export function teamStateTitle(state: string): string | undefined {
  // An empty state is a broken contract (teamTone paints it red), so it
  // gets a tooltip too, naming the empty value.
  return known("teamState", state) ? undefined : codeText(state || '""');
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
  return [...teams].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: "base" }),
  );
}

/** byLastRun sorts teams most-recently-active first, which is the order the
 *  live page wants: it answers "what is happening", and a team that ran a
 *  minute ago is more interesting than one that starts with an A.
 *
 *  A team that has never run sorts last rather than first: null is "no
 *  activity", not "infinitely old". Ties fall back to the name, so the order
 *  is stable across polls instead of wandering with whatever the server
 *  happened to emit first.
 *
 *  A COPY, as byName: the array belongs to the poller. */
export function byLastRun<T extends { name: string; last_run: string | null }>(
  teams: T[],
): T[] {
  const at = (t: T) =>
    t.last_run === null ? -Infinity : new Date(t.last_run).getTime();
  return [...teams].sort((a, b) => {
    const diff = at(b) - at(a);
    if (diff !== 0 && !Number.isNaN(diff)) return diff;
    return a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
  });
}
