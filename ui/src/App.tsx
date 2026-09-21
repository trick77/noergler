import { useEffect, useState } from "react";
import { LivePage } from "./LivePage";
import { MetricsPage } from "./MetricsPage";
import { RunsPage } from "./RunsPage";
import { TeamsPage } from "./TeamsPage";
import { PAGES, navigate, pageFromPath, type Page } from "./routing";

const TITLES: Record<Page, string> = {
  live: "Live",
  runs: "Runs",
  metrics: "Metrics",
  teams: "Teams",
};

/** The shell: a fixed header over one scrolling page. No sidebar; the four
 *  pages sit in the header as a top nav, which is where ../rongo's rail rows
 *  would have been. */
export default function App() {
  const [route, setRoute] = useState<Page>(() => pageFromPath(window.location.pathname));

  useEffect(() => {
    const onPop = () => setRoute(pageFromPath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    document.title = `${TITLES[route]} · noergler`;
  }, [route]);

  const go = (page: Page) => {
    navigate(page);
    setRoute(page);
  };

  return (
    // h-dvh, not h-screen: iOS Safari counts the collapsed toolbar into 100vh.
    <div className="grid h-dvh grid-rows-[56px_1fr]">
      <header className="flex items-center gap-4 border-b border-border bg-panel px-4 sm:gap-6 sm:px-5">
        <h1 className="font-serif text-[21px] leading-7 font-medium whitespace-nowrap text-wordmark">
          noergler
        </h1>
        <nav className="flex flex-1 items-center gap-0.5" aria-label="Pages">
          {PAGES.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => go(p)}
              aria-current={p === route ? "page" : undefined}
              className={
                "rounded-ui-sm px-3 py-1.5 text-sm transition-colors " +
                (p === route
                  ? "bg-active text-ink"
                  : "text-muted hover:bg-active hover:text-ink")
              }
            >
              {TITLES[p]}
            </button>
          ))}
        </nav>
      </header>

      <main className="min-h-0 min-w-0">
        {route === "live" && <LivePage />}
        {route === "runs" && <RunsPage />}
        {route === "metrics" && <MetricsPage />}
        {route === "teams" && <TeamsPage />}
      </main>
    </div>
  );
}
