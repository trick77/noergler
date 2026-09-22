// Hand-rolled, like ../rongo's: four pages do not need a router.

export type Page = "live" | "runs" | "metrics" | "teams" | "faq";

export const PAGES: Page[] = ["live", "runs", "metrics", "teams", "faq"];

export function pageFromPath(path: string): Page {
  const slug = path.replace(/^\/+/, "").split("/")[0];
  return (PAGES as string[]).includes(slug) ? (slug as Page) : "live";
}

export function pathFor(page: Page): string {
  return page === "live" ? "/" : `/${page}`;
}

export function navigate(page: Page) {
  const path = pathFor(page);
  // No duplicate history entry for the page you are already on.
  if (window.location.pathname === path) return;
  window.history.pushState({}, "", path);
}
