import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FaqPage } from "../FaqPage";
import { ago, money, outcomeWord, reasonLabel } from "../format";
import i18n, { browserLanguage } from "./index";
import { de } from "./de";
import { en } from "./en";

describe("browserLanguage", () => {
  it("reads German for every de-* locale", () => {
    for (const l of ["de", "de-CH", "de-DE", "de-AT", "DE-ch"]) {
      expect(browserLanguage(l)).toBe("de");
    }
  });
  it("reads English for everything else", () => {
    for (const l of ["en-US", "fr-CH", "it-CH", "rm", ""]) {
      expect(browserLanguage(l)).toBe("en");
    }
  });
});

/** leaves flattens a catalog to dotted key -> string. */
function leaves(o: object, prefix = ""): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(o)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (typeof v === "string") out[key] = v;
    else Object.assign(out, leaves(v as object, key));
  }
  return out;
}

const placeholders = (s: string) =>
  [...s.matchAll(/\{\{(\w+)\}\}|<(\w+)>/g)].map((m) => m[1] ?? m[2]).sort();

describe("catalogs", () => {
  const enLeaves = leaves(en);
  const deLeaves = leaves(de);

  it("have the same keys", () => {
    expect(Object.keys(deLeaves).sort()).toEqual(Object.keys(enLeaves).sort());
  });

  // A dropped {{count}} or <code> renders the raw value or loses the
  // markup, and only in the one language nobody tested.
  it("carry the same placeholders and tags per key", () => {
    for (const [k, v] of Object.entries(enLeaves)) {
      expect([k, placeholders(deLeaves[k])]).toEqual([k, placeholders(v)]);
    }
  });

  it("write Swiss German: double s, never eszett", () => {
    for (const [k, v] of Object.entries(deLeaves)) {
      expect([k, v.includes("ß")]).toEqual([k, false]);
    }
  });

  // Words a sweep once had to hunt out of the catalog (user, 2026-10-22):
  // Denglish verbs, colloquial contractions and slang.
  it("keep the German professional", () => {
    const banned = /\b(reviewt|gereviewt|Verdikt|kaputt|aufs|rein|raus)\b/i;
    for (const [k, v] of Object.entries(deLeaves)) {
      expect([k, banned.exec(v)?.[0] ?? null]).toEqual([k, null]);
    }
  });
});

describe("German UI", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("de");
  });
  afterEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("translates the formatter words", () => {
    const now = Date.parse("2026-10-18T12:00:00Z");
    expect(ago("2026-10-17T12:00:00Z", now)).toBe("vor 1 Tag");
    expect(ago("2026-10-15T12:00:00Z", now)).toBe("vor 3 Tagen");
    expect(money(null)).toBe("nicht bepreist");
    expect(outcomeWord("timed_out")).toBe("Zeitüberschreitung");
    expect(outcomeWord("skipped", "diff_too_large")).toBe("Diff zu gross");
  });

  // The pill's word is the cue that does not depend on colour (ok, skip and
  // fail sit at dE 7.6 under protanopia), so a failure must never read like
  // a skip. German once printed "zu gross" for both too_large and
  // diff_too_large; English printed "too_large" and "too large". Compared
  // normalised, because a reader does not see case or an underscore.
  it("never words a failure like a skip", () => {
    const norm = (s: string) =>
      s
        .toLowerCase()
        .replace(/[\s_\-.]+/g, " ")
        .trim();
    for (const lng of ["en", "de"] as const) {
      const cat = lng === "en" ? en : de;
      const failures = Object.entries(cat.outcome)
        .filter(([code]) => code !== "ok" && code !== "skipped")
        .map(([, word]) => norm(word));
      // The generic "skipped" too: it is what an unknown skip reason shows.
      for (const word of [
        ...Object.values(cat.skipWord),
        cat.outcome.skipped,
      ]) {
        expect(failures, `${lng}: skip word "${word}"`).not.toContain(
          norm(word),
        );
      }
    }
  });

  // A row written by a newer binary: the code is unknown here, so the
  // server's label shows, and the code, marked as one, if there is none.
  it("falls back for a skip reason this build does not know", () => {
    expect(reasonLabel("brand_new", "Brand new reason")).toBe(
      "Brand new reason",
    );
    expect(reasonLabel("brand_new")).toBe("Code: brand_new");
    expect(outcomeWord("skipped", "brand_new")).toBe("übersprungen");
  });

  it("renders a page with its markup intact", () => {
    render(<FaqPage />);
    expect(
      screen.getByText("Warum wurde mein PR nicht geprüft?"),
    ).toBeDefined();
    expect(screen.getByText("AGENTS.md fehlt")).toBeDefined();
    expect(screen.getByText("noergloff")).toBeDefined();
    expect(screen.getByText("/api/docs").getAttribute("href")).toBe(
      "/api/docs",
    );
  });
});
