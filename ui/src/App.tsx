import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { FaqPage } from "./FaqPage";
import { LivePage } from "./LivePage";
import { MetricsPage } from "./MetricsPage";
import { PRPage } from "./PRPage";
import { RunsPage } from "./RunsPage";
import { TeamsPage } from "./TeamsPage";
import {
  PAGES,
  navigate,
  pageFromPath,
  prRouteFromPath,
  type Page,
} from "./routing";

/** The shell: a fixed header over one scrolling page. No sidebar; the four
 *  pages sit in the header as a top nav, which is where ../rongo's rail rows
 *  would have been. */
export default function App() {
  const { t } = useTranslation();
  const [route, setRoute] = useState<Page>(() =>
    pageFromPath(window.location.pathname),
  );
  const [pr, setPR] = useState(() => prRouteFromPath(window.location.pathname));

  useEffect(() => {
    const onPop = () => {
      setRoute(pageFromPath(window.location.pathname));
      setPR(prRouteFromPath(window.location.pathname));
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    document.title =
      route === "pr"
        ? `${pr?.n == null ? t("pr.title") : t("run.title")} · noergler`
        : `${t(`nav.${route}`)} · noergler`;
  }, [route, pr, t]);

  const go = (page: Exclude<Page, "pr">) => {
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
        <nav
          className="flex flex-1 items-center gap-0.5"
          aria-label={t("nav.label")}
        >
          {PAGES.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => go(p)}
              aria-current={p === route ? "page" : undefined}
              className={
                "rounded-ui-sm px-3 py-1.5 text-sm transition-colors " +
                // A PR page belongs to Runs: highlighted, not current.
                (p === route || (p === "runs" && route === "pr")
                  ? "bg-active text-ink"
                  : "text-muted hover:bg-active hover:text-ink")
              }
            >
              {t(`nav.${p}`)}
            </button>
          ))}
          {/* An <a>, not a button: /api/docs is served by Go, outside this
              SPA, so it is a real navigation rather than a route switch. It
              is therefore not in PAGES, not in routing.ts and not in the
              embed allowlist, and it never carries aria-current. Same tab:
              it is part of the same service. */}
          <a
            href="/api/docs"
            className="rounded-ui-sm px-3 py-1.5 text-sm text-muted transition-colors hover:bg-active hover:text-ink"
          >
            {t("nav.api")}
          </a>
        </nav>
      </header>

      <main className="min-h-0 min-w-0">
        {route === "live" && <LivePage />}
        {route === "runs" && <RunsPage />}
        {/* keyed by the token: a new PR is a new page, never the previous
            one's data under the next URL while it loads (or after it
            fails). A run of the same PR is a pane inside it. */}
        {route === "pr" && pr !== null && (
          <PRPage key={pr.token} token={pr.token} n={pr.n} />
        )}
        {route === "teams" && <TeamsPage />}
        {route === "metrics" && <MetricsPage />}
        {route === "faq" && <FaqPage />}
      </main>
    </div>
  );
}
