// The small shared vocabulary. Class strings are hoisted to named consts
// rather than wrapped in components where a component would add nothing: the
// page reads as markup, and a change lands in one place.
import type { ReactNode } from "react";
import type { Tone } from "./format";

export const page = "h-full overflow-auto";
export const column = "mx-auto max-w-[900px] px-4 py-7 sm:px-6 lg:px-10";
export const h2 =
  "font-serif text-[22px] font-medium leading-tight tracking-tight text-ink sm:text-[28px]";
export const lede = "mt-1 mb-6 max-w-[66ch] text-[14.5px] text-muted";
export const eyebrow =
  "mt-7 mb-2.5 text-[11px] font-medium tracking-[.12em] text-faint uppercase";

const toneClass: Record<Tone, string> = {
  ok: "bg-online/15 text-[#82c491]",
  skip: "bg-ochre-wash text-ochre",
  fail: "bg-danger/15 text-danger-ink",
};
const dotClass: Record<Tone, string> = {
  ok: "bg-online",
  skip: "bg-ochre",
  fail: "bg-danger",
};

/** Pill always carries its word. The ok/skip/fail hues sit at dE 7.6 under
 *  protanopia, so the dot is a second cue and never the only one. */
export function Pill({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <span
      className={
        // nowrap: a two-word label ("no repos") otherwise wraps inside the
        // pill and stretches its row taller than every other one.
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium whitespace-nowrap " +
        toneClass[tone]
      }
    >
      <span aria-hidden className={"h-1.5 w-1.5 rounded-full " + dotClass[tone]} />
      {children}
    </span>
  );
}

export function TeamPill({ slug }: { slug: string }) {
  return (
    <span className="rounded-full bg-active px-2.5 py-0.5 font-mono text-[11.5px] text-ink-dim">
      {slug}
    </span>
  );
}

export function Card({
  title,
  note,
  right,
  children,
}: {
  title?: string;
  note?: string;
  right?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="mb-5 overflow-hidden rounded-ui border border-border bg-panel">
      {title && (
        <header className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 border-b border-border bg-active px-3.5 py-2.5">
          <h3 className="font-serif text-[18px] leading-tight font-medium tracking-tight text-ink">
            {title}
          </h3>
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
      <div className="-mr-px -mb-px grid grid-cols-2 sm:grid-cols-4">{children}</div>
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
      <div className="text-[11px] font-medium tracking-[.12em] text-faint uppercase">{label}</div>
      <div className="mt-1 font-serif text-[21px] leading-tight tabular-nums text-ink sm:text-[26px]">
        {value}
        {note && <small className="ml-1.5 font-sans text-[12.5px] text-muted">{note}</small>}
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
export const tdNum = td + " text-right font-mono tabular-nums text-muted";
export const tdTag = td + " font-mono text-[12.5px] whitespace-nowrap";

/** Empty is what a panel says when there is nothing, which is a real state
 *  here: an idle instance is healthy, not broken. */
export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-1.5 text-[13.5px] text-faint">{children}</p>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="mt-2.5 text-[12.5px] text-faint">{children}</p>;
}

export function Failed({ what }: { what: string }) {
  return (
    <div className="rounded-ui border border-border bg-panel px-3.5 py-3 text-[13.5px] text-muted">
      Could not load {what}. The instance may be starting, or the database may be unreachable.
    </div>
  );
}
