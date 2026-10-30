// The small shared vocabulary. Class strings are hoisted to named consts
// rather than wrapped in components where a component would add nothing: the
// page reads as markup, and a change lands in one place.
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { prUrl, type Tone } from "./format";

/** Sep is the dot between the clauses of a note or a heading.
 *
 *  An ELEMENT with CSS margins, not a string with Unicode spaces around it.
 *  U+2009 and U+2002 were both tried and neither is enough: a Unicode space
 *  is a fraction of the font size, so at the 12.5px a note renders at the
 *  gap comes out barely wider than the ordinary space it replaced, and the
 *  dot still reads as jammed against the words.
 *
 *  max() of a floor and a ratio: 7px is generous in a 12.5px note and tight
 *  beside a 28px heading, so the gap grows with the text but never collapses
 *  below what the small case needs.
 *
 *  aria-hidden: it is punctuation. A screen reader announcing "middle dot"
 *  between every clause is worse than the pause it already takes there. */
export function Sep() {
  return (
    <span aria-hidden className="mx-[max(7px,0.32em)] text-faint">
      ·
    </span>
  );
}

export const page = "h-full overflow-auto";
export const column = "mx-auto max-w-[1400px] px-4 py-7 sm:px-6 lg:px-10";
export const h2 =
  "font-serif text-[22px] font-medium leading-tight tracking-tight text-ink sm:text-[28px]";
export const lede = "mt-1 mb-6 max-w-[66ch] text-[14.5px] text-muted";
export const eyebrow =
  "mt-7 mb-2.5 text-[11px] font-medium tracking-[.12em] text-faint uppercase";

const toneClass: Record<Tone, string> = {
  ok: "bg-online/15 text-[#82c491]",
  skip: "bg-ochre-wash text-ochre",
  fail: "bg-danger/15 text-danger-ink",
  none: "bg-elevated text-ink-dim",
  // Outlined, no fill: muted clears AA on the panel (6.1:1) and on a card
  // header (5.0:1).
  quiet: "text-muted ring-1 ring-inset ring-elevated-border",
};
const dotClass: Record<Tone, string> = {
  ok: "bg-online",
  skip: "bg-ochre",
  fail: "bg-danger",
  none: "bg-muted",
  quiet: "bg-faint",
};

/** Pill always carries its word. The ok/skip/fail hues sit at dE 7.6 under
 *  protanopia, so the dot is a second cue and never the only one. none sits
 *  on the elevated fill: bg-active is already a card header, a chip and the
 *  loading skeleton. Its text is ink-dim, because muted on that fill misses
 *  AA (4.31:1). */
export function Pill({
  tone,
  title,
  children,
}: {
  tone: Tone;
  /** The long form, on hover, when the word on the pill is a short one. */
  title?: string;
  children: ReactNode;
}) {
  return (
    <span
      title={title}
      className={
        // nowrap: a two-word label ("no repos") otherwise wraps inside the
        // pill and stretches its row taller than every other one.
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium whitespace-nowrap " +
        toneClass[tone]
      }
    >
      <span
        aria-hidden
        className={"h-1.5 w-1.5 rounded-full " + dotClass[tone]}
      />
      {children}
    </span>
  );
}

/** PrTag renders a PROJECT/repo#id tag, as a link to the pull request when
 *  the instance told us its Bitbucket base. Same tab: the dashboard and
 *  Bitbucket are the same service to the person reading this.
 *
 *  A tag that does not parse, or a missing base, renders as plain text
 *  rather than a link that goes nowhere. */
export function PrTag({ tag, base }: { tag: string; base: string }) {
  const href = prUrl(tag, base);
  if (href === null) return <>{tag}</>;
  return (
    <a href={href} rel="noopener" className="hover:text-ink hover:underline">
      {tag}
    </a>
  );
}

/** TeamPill prints the team's display name and keeps the slug on hover: the
 *  slug is what the logs, the webhook path and every team= line use, so it
 *  has to stay reachable even once the page stops leading with it. Not mono
 *  any more - a name is prose, and a proportional face sets it better. */
export function TeamPill({ name, slug }: { name: string; slug: string }) {
  return (
    <span
      title={slug}
      className="rounded-full bg-active px-2.5 py-0.5 text-[11.5px] whitespace-nowrap text-ink-dim"
    >
      {name}
    </span>
  );
}

export function Card({
  id,
  title,
  hint,
  titleMark,
  afterTitle,
  note,
  right,
  className = "",
  children,
}: {
  /** An anchor id, for a card a path names (the Teams page's cards). */
  id?: string;
  title?: string;
  /** Hover text on the title, e.g. the slug behind a display name. */
  hint?: string;
  /** The title rendered with marks (a search hit); `title` stays the
   *  plain text for the heading's name. */
  titleMark?: ReactNode;
  /** A control right after the title, e.g. a copy-link glyph. */
  afterTitle?: ReactNode;
  note?: ReactNode;
  right?: ReactNode;
  /** Extra classes on the section, e.g. the deep-link ring. */
  className?: string;
  children: ReactNode;
}) {
  return (
    <section
      id={id}
      className={
        "group/card mb-5 overflow-hidden rounded-ui border border-border bg-panel " +
        className
      }
    >
      {/* A card with only `right` still gets the header bar: a control
          belongs on the rule above the table, not floating in the body. */}
      {(title || right) && (
        <header className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 border-b border-border bg-active px-3.5 py-2.5">
          {title && (
            <h3
              title={hint}
              className="font-serif text-[18px] leading-tight font-medium tracking-tight text-ink"
            >
              {titleMark ?? title}
            </h3>
          )}
          {afterTitle}
          {note && <span className="text-[12.5px] text-muted">{note}</span>}
          {right && <span className="ml-auto">{right}</span>}
        </header>
      )}
      <div className="px-3.5 py-3">{children}</div>
    </section>
  );
}

/** Tiles is the hairline stat grid: negative margins on the grid and a
 *  border on every cell, so adjacent cells share one rule instead of
 *  drawing two. */
export function Tiles({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-hidden rounded-ui border border-border bg-panel">
      <div className="-mr-px -mb-px grid grid-cols-2 sm:grid-cols-4">
        {children}
      </div>
    </div>
  );
}

export function Tile({
  label,
  value,
  note,
  children,
}: {
  label: string;
  value: ReactNode;
  note?: string;
  children?: ReactNode;
}) {
  return (
    <div className="border-r border-b border-border px-3.5 py-3">
      <div className="text-[11px] font-medium tracking-[.12em] text-faint uppercase">
        {label}
      </div>
      <div className="mt-1 font-serif text-[21px] leading-tight tabular-nums text-ink sm:text-[26px]">
        {value}
        {note && (
          <small className="ml-1.5 font-sans text-[12.5px] text-muted">
            {note}
          </small>
        )}
      </div>
      {children}
    </div>
  );
}

export const table = "w-full border-separate border-spacing-0 text-[13.5px]";
export const th =
  "border-b border-border pb-1.5 pr-2.5 text-left text-[11px] font-medium tracking-[.1em] text-faint uppercase";
export const thNum = th + " text-right";
export const td = "border-b border-border-soft py-1.5 pr-2.5 text-ink-dim";
// nowrap: "12m ago" or "1.2s" split over two lines reads as two values.
export const tdNum =
  td + " text-right font-mono tabular-nums text-muted whitespace-nowrap";
export const tdTag = td + " font-mono text-[12.5px] whitespace-nowrap";

/** Empty is what a panel says when there is nothing, which is a real state
 *  here: an idle instance is healthy, not broken. */
export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-1.5 text-[13.5px] text-faint">{children}</p>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="mt-2.5 text-[12.5px] text-faint">{children}</p>;
}

/** Failed takes `what` already translated, in the case the sentence needs
 *  ("the live panel", "das Live-Panel"). */
export function Failed({ what }: { what: string }) {
  const { t } = useTranslation();
  return (
    <div className="rounded-ui border border-border bg-panel px-3.5 py-3 text-[13.5px] text-muted">
      {t("common.failed", { what })}
    </div>
  );
}
