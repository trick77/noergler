import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import { de } from "./de";
import { en, type Catalog } from "./en";

// Two UI languages, picked from the browser and nothing else: any de-*
// locale reads German, everything else English. No switch and no stored
// choice (decided): the dashboard has no login and no profile to keep one in.
export type UiLanguage = "en" | "de";

export function browserLanguage(
  nav: string = (typeof navigator !== "undefined" && navigator.language) || "",
): UiLanguage {
  return nav.toLowerCase().startsWith("de") ? "de" : "en";
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
  lng: browserLanguage(),
  fallbackLng: "en",
  interpolation: { escapeValue: false },
  initAsync: false,
});

if (typeof document !== "undefined") {
  document.documentElement.lang = i18n.language;
}

export default i18n;
