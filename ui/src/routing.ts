// Hand-rolled, like ../rongo's: a handful of pages do not need a router.

export type Page = "live" | "runs" | "teams" | "metrics" | "faq" | "run";

/** PAGES is the nav. "run" is not in it: a run detail page is reached only
 *  from the PR's summary comment, and it highlights Runs while open. */
export const PAGES: Exclude<Page, "run">[] = [
  "live",
  "runs",
  "teams",
  "metrics",
  "faq",
];

// Mirrors runPath in backend/web/embed.go: a link token, 22 base64url
// characters. The page's only key, handed out in the PR's summary comment;
// the dashboard never links to it.
const RUN_PATH = /^\/runs\/([A-Za-z0-9_-]{22})$/;

export function pageFromPath(path: string): Page {
  if (RUN_PATH.test(path)) return "run";
  const slug = path.replace(/^\/+/, "").split("/")[0];
  return (PAGES as string[]).includes(slug) ? (slug as Page) : "live";
}

/** runTokenFromPath is the link token of a run detail path, null
 *  otherwise. */
export function runTokenFromPath(path: string): string | null {
  const m = RUN_PATH.exec(path);
  return m === null ? null : m[1];
}

export function pathFor(page: Exclude<Page, "run">): string {
  return page === "live" ? "/" : `/${page}`;
}

export function navigate(page: Exclude<Page, "run">) {
  navigateTo(pathFor(page));
}

/** onNav is an in-app link's click handler: a plain click stays in the SPA,
 *  a modified one (new tab, new window) is left to the browser. */
export function onNav(path: string) {
  return (e: {
    metaKey: boolean;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    button: number;
    preventDefault: () => void;
  }) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0)
      return;
    e.preventDefault();
    navigateTo(path);
  };
}

/** navigateTo pushes any app path. No duplicate history entry for the page
 *  you are already on. */
export function navigateTo(path: string) {
  if (window.location.pathname === path) return;
  window.history.pushState({}, "", path);
  // Tell the shell, which otherwise only hears back/forward (popstate).
  window.dispatchEvent(new PopStateEvent("popstate"));
}
