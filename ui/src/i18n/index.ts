import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import { de } from "./de";
import { en, type Catalog } from "./en";

// Two UI languages. A stored choice (the header's switch) wins; without one
// the browser decides: any de-* locale reads German, everything else
// English. Same rule and storage key as ../riptide, so both read alike.
// Only words follow the language; numbers and dates stay Swiss (format.ts).
export type UiLanguage = "en" | "de";

const STORAGE_KEY = "lang";

export function browserLanguage(
  nav: string = (typeof navigator !== "undefined" && navigator.language) || "",
): UiLanguage {
  return nav.toLowerCase().startsWith("de") ? "de" : "en";
}

// Storage can be blocked (private mode, a sandboxed frame) or hold junk
// from an older build: either reads as no choice.
export function storedLanguage(): UiLanguage | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "de" || v === "en" ? v : null;
  } catch {
    return null;
  }
}

export function initialLanguage(): UiLanguage {
  return storedLanguage() ?? browserLanguage();
}

// setLanguage is the switch: the page re-renders in lng, and the choice
// outlives the tab. A blocked storage still switches, for this page only.
export function setLanguage(lng: UiLanguage) {
  try {
    localStorage.setItem(STORAGE_KEY, lng);
  } catch {
    // not kept; the switch still applies
  }
  void i18n.changeLanguage(lng);
}

declare module "i18next" {
  interface CustomTypeOptions {
    resources: { translation: Catalog };
  }
}

// Synchronous, at import: the first render already has its language, so
// there is no flash of English for a German reader.
void i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, de: { translation: de } },
  lng: initialLanguage(),
  fallbackLng: "en",
  interpolation: { escapeValue: false },
  initAsync: false,
});

const syncLang = (lng: string) => {
  if (typeof document !== "undefined") document.documentElement.lang = lng;
};
syncLang(i18n.language);
i18n.on("languageChanged", syncLang);

// A switch in another tab: follow it, or this tab's button would offer the
// language the other tab already stored.
function followOtherTab(e: StorageEvent) {
  if (e.key !== STORAGE_KEY) return;
  const lng = storedLanguage();
  if (lng !== null && lng !== i18n.language) void i18n.changeLanguage(lng);
}
if (typeof window !== "undefined") {
  window.addEventListener("storage", followOtherTab);
}

export default i18n;
