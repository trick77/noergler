// Hand-rolled, like ../rongo's: a handful of pages do not need a router.

export type Page = "live" | "runs" | "metrics" | "teams" | "faq" | "run";

/** PAGES is the nav. "run" is not in it: a run detail page is reached from
 *  a row, never from the header, and it highlights Runs while open. */
export const PAGES: Exclude<Page, "run">[] = [
  "live",
  "runs",
  "metrics",
  "teams",
  "faq",
];

// Mirrors runPath in backend/web/embed.go: a positive id, no leading zero.
const RUN_PATH = /^\/runs\/([1-9][0-9]{0,18})$/;

export function pageFromPath(path: string): Page {
  if (RUN_PATH.test(path)) return "run";
  const slug = path.replace(/^\/+/, "").split("/")[0];
  return (PAGES as string[]).includes(slug) ? (slug as Page) : "live";
}

/** runIdFromPath is the attempt id of a run detail path, null otherwise. */
export function runIdFromPath(path: string): number | null {
  const m = RUN_PATH.exec(path);
  return m === null ? null : Number(m[1]);
}

export function pathFor(page: Exclude<Page, "run">): string {
  return page === "live" ? "/" : `/${page}`;
}

/** runPath is the run detail page of one attempt. */
export function runPath(id: number): string {
  return `/runs/${id}`;
}

export function navigate(page: Exclude<Page, "run">) {
  navigateTo(pathFor(page));
}

/** navigateTo pushes any app path. No duplicate history entry for the page
 *  you are already on. */
export function navigateTo(path: string) {
  if (window.location.pathname === path) return;
  window.history.pushState({}, "", path);
  // Tell the shell, which otherwise only hears back/forward (popstate).
  window.dispatchEvent(new PopStateEvent("popstate"));
}
