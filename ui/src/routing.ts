// Hand-rolled, like ../rongo's: a handful of pages do not need a router.

export type Page = "live" | "runs" | "teams" | "metrics" | "faq" | "pr";

/** PAGES is the nav. "pr" is not in it: a PR page and its runs are reached
 *  only from the PR's summary comment, and highlight Runs while open. */
export const PAGES: Exclude<Page, "pr">[] = [
  "live",
  "runs",
  "teams",
  "metrics",
  "faq",
];

// Mirrors prPath in backend/web/embed.go: a link token, 22 base64url
// characters, then optionally a run's number. The token is the pages' only
// key, handed out in the PR's summary comment; the rest of the dashboard
// never links to it.
const PR_PATH = /^\/prs\/([A-Za-z0-9_-]{22})(?:\/runs\/([1-9][0-9]{0,8}))?$/;

/** PRRoute is a PR page (n null) or run n of it. */
export interface PRRoute {
  token: string;
  n: number | null;
}

// Mirrors teamPath in backend/web/embed.go: one team on the Teams page, the
// slug as config.TeamSlugRE shapes it. The page itself, scrolled to that
// team with its configuration open; the link a team card's # copies.
const TEAM_PATH = /^\/teams\/([a-z0-9][a-z0-9-]*)$/;

/** teamSlugFromPath is the team a Teams path names, null on /teams itself. */
export function teamSlugFromPath(path: string): string | null {
  const m = TEAM_PATH.exec(path);
  return m === null ? null : m[1];
}

/** teamPath is one team's link on the Teams page. */
export function teamPath(slug: string): string {
  return `/teams/${slug}`;
}

export function pageFromPath(path: string): Page {
  if (PR_PATH.test(path)) return "pr";
  const slug = path.replace(/^\/+/, "").split("/")[0];
  return (PAGES as string[]).includes(slug) ? (slug as Page) : "live";
}

/** prRouteFromPath is the PR page or run a path names, null otherwise. */
export function prRouteFromPath(path: string): PRRoute | null {
  const m = PR_PATH.exec(path);
  if (m === null) return null;
  return { token: m[1], n: m[2] === undefined ? null : Number(m[2]) };
}

/** prPath is the PR page's path, or run n's under it. */
export function prPath(token: string, n?: number): string {
  return n === undefined ? `/prs/${token}` : `/prs/${token}/runs/${n}`;
}

export function pathFor(page: Exclude<Page, "pr">): string {
  return page === "live" ? "/" : `/${page}`;
}

export function navigate(page: Exclude<Page, "pr">) {
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
