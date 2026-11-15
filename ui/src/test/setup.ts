import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import i18n from "../i18n";

// The language switch keeps its choice in localStorage and switches the one
// i18n instance: a test that clicks it must not hand the next test a German
// page.
afterEach(async () => {
  localStorage.clear();
  await i18n.changeLanguage("en");
});
