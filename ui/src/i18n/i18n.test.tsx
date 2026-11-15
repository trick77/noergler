import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FaqPage } from "../FaqPage";
import {
  ago,
  dayMonth,
  money,
  monthName,
  num,
  outcomeWord,
  reasonLabel,
  when,
} from "../format";
import { LanguageSwitch } from "../ui";
import i18n, {
  browserLanguage,
  initialLanguage,
  setLanguage,
  storedLanguage,
} from "./index";
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

describe("language choice", () => {
  afterEach(async () => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    await i18n.changeLanguage("en");
  });

  const browser = (language: string) =>
    vi.stubGlobal("navigator", { ...navigator, language });
  const blockStorage = (method: "getItem" | "setItem") =>
    vi.spyOn(Storage.prototype, method).mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

  it("lets a stored choice beat the browser", () => {
    browser("de-CH");
    localStorage.setItem("lang", "en");
    expect(initialLanguage()).toBe("en");
    browser("en-US");
    localStorage.setItem("lang", "de");
    expect(initialLanguage()).toBe("de");
  });

  it("falls back to the browser without a choice", () => {
    browser("de-CH");
    expect(storedLanguage()).toBeNull();
    expect(initialLanguage()).toBe("de");
  });

  it("reads junk from an older build as no choice", () => {
    browser("de-AT");
    localStorage.setItem("lang", "fr");
    expect(storedLanguage()).toBeNull();
    expect(initialLanguage()).toBe("de");
  });

  it("reads blocked storage as no choice", () => {
    browser("de-CH");
    localStorage.setItem("lang", "en");
    blockStorage("getItem");
    expect(storedLanguage()).toBeNull();
    expect(initialLanguage()).toBe("de");
  });

  it("keeps the switched language and names it on <html lang>", () => {
    setLanguage("de");
    expect(localStorage.getItem("lang")).toBe("de");
    expect(i18n.language).toBe("de");
    expect(document.documentElement.lang).toBe("de");
    setLanguage("en");
    expect(localStorage.getItem("lang")).toBe("en");
    expect(document.documentElement.lang).toBe("en");
  });

  it("still switches when storage refuses the choice", () => {
    blockStorage("setItem");
    setLanguage("de");
    expect(i18n.language).toBe("de");
    expect(document.documentElement.lang).toBe("de");
  });

  it("switches the page from the header button", () => {
    render(<LanguageSwitch />);
    const button = screen.getByRole("button", {
      name: "Language: English. Switch to German",
    });
    expect(button.textContent).toBe("EN");
    act(() => {
      fireEvent.click(button);
    });
    expect(button.textContent).toBe("DE");
    expect(button.getAttribute("aria-label")).toBe(
      "Sprache: Deutsch. Auf Englisch wechseln",
    );
    expect(localStorage.getItem("lang")).toBe("de");
  });

  it("follows a switch made in another tab", () => {
    localStorage.setItem("lang", "de");
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", { key: "lang" }));
    });
    expect(i18n.language).toBe("de");
    localStorage.setItem("other", "en");
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", { key: "other" }));
    });
    expect(i18n.language).toBe("de");
  });
});

// Counts and dates stay Swiss in either language; only a month name is a
// word and follows it (user, 2026-11-15). One date shape in both: a
// dateStyle once gave de-CH a two-digit year and en-CH a four-digit one.
describe("formats", () => {
  afterEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("are Swiss in both languages, with the month in the reader's", async () => {
    const at = new Date(2026, 9, 2, 14, 5).toISOString();
    for (const [lng, month, day] of [
      ["en", "October 2026", "2 Oct"],
      ["de", "Oktober 2026", "2. Okt."],
    ] as const) {
      await i18n.changeLanguage(lng);
      expect(num(1540)).toBe("1'540");
      expect(when(at)).toBe("02.10.2026, 14:05");
      expect(when(at, true)).toBe("02.10.2026, 14:05:00");
      expect(monthName("2026-10-01T00:00:00+02:00")).toBe(month);
      expect(dayMonth("2026-10-02")).toBe(day);
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
  // Denglish verbs, colloquial contractions and slang. A Denglish participle
  // ends in -ed, never -t: geposted, gemerged (user, 2026-10-26).
  it("keep the German professional", () => {
    const banned =
      /\b(reviewt|gereviewt|gepostet\w*|gemergt\w*|Verdikt|kaputt|aufs|rein|raus)\b/i;
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
    expect(outcomeWord("skipped", "brand_new")).toBe("Übersprungen");
  });

  it("renders a page with its markup intact", () => {
    render(<FaqPage />);
    expect(
      screen.getByText("Warum wurde mein PR nicht geprüft?"),
    ).toBeDefined();
    expect(screen.getByText("AGENTS.md fehlt")).toBeDefined();
    expect(screen.getByText("noergloff")).toBeDefined();
  });
});
