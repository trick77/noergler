import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FaqPage } from "./FaqPage";
import { LivePage } from "./LivePage";
import { MetricsPage } from "./MetricsPage";
import { RunsPage } from "./RunsPage";
import { TeamsPage } from "./TeamsPage";
import type { Live, Metrics, Run, Team } from "./api";

/** serve answers each dashboard path from a table, so a page under test
 *  fetches exactly what it would in the browser. */
function serve(table: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const path = url.replace("/api/dashboard/", "").split("?")[0];
      if (!(path in table)) {
        return { ok: false, status: 500, json: async () => ({}) };
      }
      return { ok: true, status: 200, json: async () => table[path] };
    }),
  );
}

const live: Live = {
  pool_capacity: 6,
  pool_per_team: 2,
  staged: 2,
  depth: 1,
  running: [
    {
      tag: "PAY/ledger#1",
      team: "payments",
      team_name: "Payments",
      kind: "review",
      since: new Date().toISOString(),
    },
  ],
  waiting: [
    {
      tag: "PAY/ledger#2",
      team: "payments",
      team_name: "Payments",
      since: new Date().toISOString(),
    },
  ],
  // Only teams that have run: the server leaves out disabled and idle ones.
  // Named so that name order (Ausgaben, Mobile, Search) and last-run order
  // (Search, Mobile, Ausgaben) differ: the page's own sort has to put them
  // right.
  teams: [
    {
      slug: "payments",
      name: "Ausgaben",
      prs: 4,
      last_reviewed: new Date(Date.now() - 600_000).toISOString(),
      last_run: new Date(Date.now() - 600_000).toISOString(),
      last_outcome: "ok",
    },
    // Failed and never reviewed: last run set, last reviewed "never".
    {
      slug: "mobile",
      name: "Mobile",
      prs: 1,
      last_reviewed: null,
      last_run: new Date(Date.now() - 300_000).toISOString(),
      last_outcome: "timed_out",
    },
    // A skip newer than its last success: the case one time for both hid.
    {
      slug: "search",
      name: "Search",
      prs: 2,
      last_reviewed: new Date(Date.now() - 7_200_000).toISOString(),
      last_run: new Date(Date.now() - 60_000).toISOString(),
      last_outcome: "skipped",
      last_reason: "head_unchanged",
      last_reason_label: "HEAD unchanged since last review",
    },
  ],
  bitbucket_url: "https://bitbucket.example.com",
};

const metrics: Metrics = {
  since: "2026-09-01T00:00:00",
  until: "2026-10-01T00:00:00",
  window: "month",
  lead_since: "2026-08-26T00:00:00",
  totals: {
    runs: 12,
    prompt_tokens: 1000,
    cached_tokens: 0,
    completion_tokens: 100,
    findings_posted: 5,
    cost_usd: "1.250",
    unpriced_runs: 3,
    lines_added: 1200,
    lines_removed: 340,
    lines_uncounted_runs: 0,
  },
  prs: { reviewed: 4, seen: 7 },
  by_team: [
    {
      team: "payments",
      team_name: "Payments",
      runs: 12,
      prompt_tokens: 1000,
      cached_tokens: 0,
      completion_tokens: 100,
      findings_posted: 5,
      cost_usd: "1.250",
      unpriced_runs: 3,
      lines_added: 1200,
      lines_removed: 340,
      lines_uncounted_runs: 0,
    },
  ],
  daily: [
    {
      day: "2026-09-02",
      team: "payments",
      runs: 2,
      findings_posted: 5,
      cost_usd: "0.500",
      unpriced_runs: 0,
      lines: 1540,
    },
  ],
  daily_attempts: [
    { day: "2026-09-02", outcome: "ok", count: 2 },
    { day: "2026-09-02", outcome: "skipped", count: 4 },
  ],
  breakdown: [
    { outcome: "ok", count: 2 },
    {
      outcome: "skipped",
      reason: "head_unchanged",
      label: "HEAD unchanged since last review",
      count: 4,
    },
    { outcome: "error", count: 1 },
  ],
};

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("LivePage", () => {
  it("draws the whole pool, not just what is running", async () => {
    serve({ live });
    render(<LivePage />);

    // Six slots, because the pool is six wide: a panel showing one row per
    // running review would report a parallel instance as serial.
    const slots = await screen.findByLabelText("2 of 6 slots busy");
    expect(slots.children).toHaveLength(6);
    expect(await screen.findByText("PAY/ledger#1")).toBeDefined();
    expect(await screen.findByText("PAY/ledger#2")).toBeDefined();
  });

  it("names a team, with the slug on hover", async () => {
    serve({ live });
    render(<LivePage />);

    // The display name leads; the slug stays reachable on hover, because
    // it is what team= and the webhook path use.
    const row = await screen.findByText("Mobile");
    expect(row.getAttribute("title")).toBe("mobile");
  });

  // Most recently run first, skips and failures included: the page answers
  // "what is happening", so a team that ran a minute ago outranks one whose
  // name starts with an A.
  it("orders teams by last run, not by name", async () => {
    serve({ live });
    render(<LivePage />);

    // Scoped to the Teams table: the running and queue cards carry team
    // pills with the same title attribute.
    const table = (await screen.findByText("Mobile")).closest("table");
    const names = Array.from(
      table?.querySelectorAll("tbody tr td:first-child") ?? [],
    ).map((el) => el.textContent);
    expect(names).toEqual(["Search", "Mobile", "Ausgaben"]);
  });

  // "Last run" once showed the last SUCCESS, so a team whose latest run was
  // skipped or failed looked idle since its last review. The outcome gets
  // its own column: a pill trailing a time reads as a label on the time.
  it("splits last reviewed from last run, outcome in its own column", async () => {
    serve({ live });
    render(<LivePage />);

    const row = (await screen.findByText("Search")).closest("tr");
    const cells = Array.from(row?.querySelectorAll("td") ?? []).map(
      (el) => el.textContent,
    );
    // No PR count: it says nothing about whether the team is live.
    expect(cells).toEqual(["Search", "Unchanged", "2h ago", "1m ago"]);
    expect(screen.getByText("Unchanged").getAttribute("title")).toBe(
      "HEAD unchanged since last review",
    );

    const mobile = (await screen.findByText("Mobile")).closest("tr");
    const mobileCells = Array.from(mobile?.querySelectorAll("td") ?? []).map(
      (el) => el.textContent,
    );
    expect(mobileCells).toEqual(["Mobile", "Timed out", "never", "5m ago"]);
  });

  // State is configuration, not live: the Teams page carries it.
  it("does not carry the state column", async () => {
    serve({ live });
    render(<LivePage />);

    await screen.findByText("Ausgaben");
    expect(screen.queryByText("State")).toBeNull();
    expect(screen.queryByText("ready")).toBeNull();
  });

  it("says so when no team has run", async () => {
    serve({ live: { ...live, teams: [] } });
    render(<LivePage />);
    expect(await screen.findByText("No runs yet.")).toBeDefined();
  });

  // What a team claims is configuration read once at boot, not something
  // that moves while you watch. It belongs on the teams page, which shows
  // the claims themselves rather than a sentinel word for them.
  it("does not carry the scope column", async () => {
    serve({ live });
    render(<LivePage />);

    await screen.findByText("Ausgaben");
    expect(screen.queryByText("Scope")).toBeNull();
    expect(screen.queryByText("project")).toBeNull();
  });

  it("says so when the panel cannot be loaded", async () => {
    serve({});
    render(<LivePage />);
    expect(await screen.findByText(/Could not load/)).toBeDefined();
  });

  // Once the panel has loaded, a dead instance kept its last running rows
  // and their elapsed clocks counting up: a review "stuck" for 40 minutes.
  it("says so when the refresh fails, and stops the clocks", async () => {
    let calls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        calls += 1;
        if (calls > 1)
          return { ok: false, status: 503, json: async () => ({}) };
        return { ok: true, status: 200, json: async () => live };
      }),
    );
    render(<LivePage />);
    await screen.findByText("PAY/ledger#1");

    await vi.advanceTimersByTimeAsync(3000);
    expect(await screen.findByText(/Refresh failing/)).toBeDefined();
    const elapsed = screen.getByText("PAY/ledger#1").closest("tr")
      ?.lastElementChild?.textContent;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(
      screen.getByText("PAY/ledger#1").closest("tr")?.lastElementChild
        ?.textContent,
    ).toBe(elapsed);
  });
});

describe("RunsPage", () => {
  const runs: Run[] = [
    {
      id: 1,
      tag: "PAY/ledger#1",
      team: "payments",
      team_name: "Payments",
      kind: "auto",
      outcome: "ok",
      elapsed_ms: 72400,
      findings: 6,
      cost_usd: "0.218",
      created_at: new Date().toISOString(),
    },
    {
      id: 2,
      tag: "PAY/ledger#2",
      team: "payments",
      team_name: "Payments",
      kind: "auto",
      outcome: "ok",
      elapsed_ms: 41000,
      findings: 0,
      cost_usd: null,
      created_at: new Date().toISOString(),
    },
    {
      id: 3,
      tag: "PAY/ledger#3",
      team: "payments",
      team_name: "Payments",
      kind: "auto",
      outcome: "skipped",
      reason: "head_unchanged",
      reason_label: "HEAD unchanged since last review",
      elapsed_ms: null,
      findings: null,
      cost_usd: null,
      created_at: new Date().toISOString(),
    },
    {
      id: 4,
      tag: "PAY/ledger#4",
      team: "payments",
      team_name: "Payments",
      kind: "auto",
      outcome: "timed_out",
      elapsed_ms: 300000,
      findings: null,
      cost_usd: null,
      created_at: new Date().toISOString(),
    },
  ];

  it("shows failures and skips beside the successes", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);

    expect(await screen.findByText("Timed out")).toBeDefined();
    // A skip's pill names its reason, not the bare outcome.
    expect(await screen.findByText("Unchanged")).toBeDefined();
    expect(screen.getAllByText("OK")).toHaveLength(2);
  });

  it("keeps an unpriced run unpriced", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);

    expect(await screen.findByText("$0.22")).toBeDefined();
    expect(await screen.findByText("unpriced")).toBeDefined();
    // The failures have no cost at all, which is a dash, not $0.00.
    expect(screen.queryByText("$0.00")).toBeNull();
  });

  // The page is the latest 50 runs and nothing wider (user, 2026-11-10):
  // one feed, no /metrics, and the tiles count the rows on screen.
  it("counts the feed's own rows, from one request", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);

    await screen.findByText("PAY/ledger#1");
    // The label's sibling is the value; the heading and the filter pills
    // carry the same words.
    const tile = (label: string) =>
      screen
        .getAllByText(label)
        .map((el) => el.nextElementSibling)
        .find((el) => el?.className.includes("font-serif"))?.textContent;
    // No Runs tile: it would read 50 on any busy instance.
    expect(tile("Runs")).toBeUndefined();
    expect(tile("Reviewed")).toBe("2");
    expect(tile("Skipped")).toBe("1");
    expect(tile("Failed")).toBe("1");
    const calls = vi.mocked(fetch).mock.calls.map(([u]) => String(u));
    expect(calls.every((u) => u.endsWith("/runs?limit=50"))).toBe(true);
  });

  // "Failed 1" on the tile is the one row the filter shows: same rows,
  // narrowed in the browser rather than a second, wider query.
  it("filters the same rows the tiles count", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);
    await screen.findByText("PAY/ledger#1");

    fireEvent.click(screen.getByRole("button", { name: "Failed" }));
    expect(screen.getByText("PAY/ledger#4")).toBeDefined();
    expect(screen.queryByText("PAY/ledger#1")).toBeNull();
    expect(
      screen
        .getByRole("button", { name: "Failed" })
        .getAttribute("aria-pressed"),
    ).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "Skipped" }));
    expect(screen.getByText("PAY/ledger#3")).toBeDefined();
    expect(screen.queryByText("PAY/ledger#4")).toBeNull();
  });

  it("says a filter found nothing among the latest 50", async () => {
    serve({ runs: { runs: runs.filter((r) => r.outcome === "ok") } });
    render(<RunsPage />);
    await screen.findByText("PAY/ledger#1");

    fireEvent.click(screen.getByRole("button", { name: "Failed" }));
    expect(
      screen.getByText("No failed runs among the latest 50."),
    ).toBeDefined();
  });

  // The skip breakdown is the feed's skips by reason, not a window's.
  it("breaks the feed's skips down by reason", async () => {
    const skip = runs[2];
    serve({
      runs: {
        runs: [
          ...runs,
          { ...skip, id: 5 },
          { ...skip, id: 6, reason: "draft", reason_label: "Draft PR" },
        ],
      },
    });
    render(<RunsPage />);

    const head = (
      await screen.findAllByText("HEAD unchanged since last review")
    ).find((el) => el.tagName === "TD");
    expect(head?.nextElementSibling?.textContent).toBe("2");
  });

  // A refresh that starts failing keeps the rows but says how old they
  // are, and the "when" column stops counting on: stale rows must not read
  // as a live page.
  it("says so when the refresh fails", async () => {
    let calls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        calls += 1;
        if (calls > 1)
          return { ok: false, status: 503, json: async () => ({}) };
        return { ok: true, status: 200, json: async () => ({ runs }) };
      }),
    );
    render(<RunsPage />);
    await screen.findByText("PAY/ledger#1");
    expect(screen.queryByText(/Refresh failing/)).toBeNull();

    await vi.advanceTimersByTimeAsync(10_000);
    expect(await screen.findByText(/Refresh failing/)).toBeDefined();
    expect(screen.getByText("PAY/ledger#1")).toBeDefined();
  });

  // The ledes are gone (user, 2026-11-10: "ai slop").
  it("carries no lede", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);
    await screen.findByText("PAY/ledger#1");
    expect(screen.queryByText(/What noergler did/)).toBeNull();
  });

  // By count, ties by label: insertion order is newest first, so equal
  // counts swapped places as the feed moved. The label shows once, in the
  // breakdown: on the row it is the pill's tooltip.
  it("ranks the skip reasons by count, ties by label", async () => {
    const skip = runs[2];
    const as = (id: number, reason: string, reason_label: string) => ({
      ...skip,
      id,
      reason,
      reason_label,
    });
    serve({
      runs: {
        runs: [
          as(10, "zeta", "Zeta"),
          as(11, "alpha", "Alpha"),
          as(12, "zeta", "Zeta"),
          as(13, "alpha", "Alpha"),
          as(14, "mid", "Mid"),
          as(15, "mid", "Mid"),
          as(16, "mid", "Mid"),
        ],
      },
    });
    render(<RunsPage />);

    const table = (await screen.findByText("Reason")).closest("table");
    const rows = Array.from(table?.querySelectorAll("tbody tr") ?? []).map(
      (tr) => tr.textContent,
    );
    expect(rows).toEqual(["Mid3", "Alpha2", "Zeta2"]);
  });

  it("shows a skip's label once, in the breakdown", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);
    expect(
      await screen.findAllByText("HEAD unchanged since last review"),
    ).toHaveLength(1);
  });

  // The long label beside the pill wrapped the Outcome cell into three or
  // four lines. The pill says it in a word and keeps the label on hover, so
  // the row still knows why without sending the reader to the logs.
  it("says why a run was skipped in the pill, with the label on hover", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);

    const pill = await screen.findByText("Unchanged");
    expect(pill.getAttribute("title")).toBe("HEAD unchanged since last review");
  });

  it("links a PR tag to Bitbucket", async () => {
    serve({
      runs: { runs, bitbucket_url: "https://bitbucket.example.com" },
    });
    render(<RunsPage />);

    const link = await screen.findByText("PAY/ledger#1");
    expect(link.getAttribute("href")).toBe(
      "https://bitbucket.example.com/projects/PAY/repos/ledger/pull-requests/1",
    );
  });

  // BITBUCKET_URL unset: the tag is still the row's identity, so it stays
  // readable rather than becoming a link that goes nowhere.
  it("leaves the tag as text when the instance has no Bitbucket base", async () => {
    serve({ runs: { runs, bitbucket_url: "" } });
    render(<RunsPage />);

    const tag = await screen.findByText("PAY/ledger#1");
    expect(tag.getAttribute("href")).toBeNull();
  });
});

describe("MetricsPage", () => {
  it("reports unpriced runs beside the cost, never inside it", async () => {
    serve({ metrics });
    render(<MetricsPage />);

    // Twice on purpose: the headline tile and the by-team row. The point
    // is that the figure is the priced sum in both places.
    expect(await screen.findAllByText("$1.25")).toHaveLength(2);
    expect(await screen.findByText("Unpriced")).toBeDefined();
    // The unpriced count stands on its own rather than being folded in.
    expect(await screen.findAllByText("3")).not.toHaveLength(0);
  });

  // A fresh instance has run nothing, which is not the same as having run
  // something the gateway declined to price. "unpriced" there would claim a
  // fact about a run that never happened.
  it("shows a dash rather than unpriced before anything has run", async () => {
    serve({
      metrics: {
        ...metrics,
        totals: {
          ...metrics.totals,
          runs: 0,
          cost_usd: null,
          unpriced_runs: 0,
        },
        prs: { reviewed: 0, seen: 0 },
        by_team: [],
        daily: [],
        daily_attempts: [],
        breakdown: [],
      },
    });
    render(<MetricsPage />);

    await waitFor(() => expect(screen.getByText("Cost")).toBeDefined());
    expect(screen.queryByText("unpriced")).toBeNull();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  // Reviewed counts PRs, reviewed out of seen. No runs line beneath it
  // (user, 2026-11-10); the team table still says Runs.
  it("shows reviewed PRs out of seen, without a runs line", async () => {
    serve({ metrics });
    render(<MetricsPage />);

    const value = (await screen.findByText("Reviewed")).nextElementSibling;
    expect(value?.textContent).toBe("4 / 7PRs");
    expect(screen.getByTitle("4 of 7 PRs seen got reviewed")).toBeDefined();
    expect(screen.queryByText("12 runs")).toBeNull();
    expect(screen.getByRole("columnheader", { name: "Runs" })).toBeDefined();
  });

  // + and − as one figure, the split on hover, in the tile and the team
  // row; a dash before anything ran.
  it("shows lines reviewed as one figure with the split on hover", async () => {
    serve({ metrics });
    render(<MetricsPage />);

    const value = (await screen.findByText("Lines reviewed"))
      .nextElementSibling;
    expect(value?.textContent).toBe((1540).toLocaleString());
    const split = `+${(1200).toLocaleString()} / −340: lines sent to the model, each counted once per PR`;
    expect(screen.getAllByTitle(split)).toHaveLength(2);
    expect(screen.getByRole("columnheader", { name: "Lines" })).toBeDefined();
  });

  it("shows a dash for lines reviewed before anything has run", async () => {
    serve({
      metrics: {
        ...metrics,
        totals: {
          ...metrics.totals,
          runs: 0,
          lines_added: 0,
          lines_removed: 0,
        },
        by_team: [],
        daily: [],
      },
    });
    render(<MetricsPage />);

    const value = (await screen.findByText("Lines reviewed"))
      .nextElementSibling;
    expect(value?.textContent).toBe("—");
  });

  // Runs from before the figure existed add no lines: some of them is a
  // note beside the number, all of them a dash, never a silent low figure.
  it("says when older runs are not in lines reviewed", async () => {
    serve({
      metrics: {
        ...metrics,
        totals: { ...metrics.totals, lines_uncounted_runs: 9 },
      },
    });
    render(<MetricsPage />);
    const value = (await screen.findByText("Lines reviewed"))
      .nextElementSibling;
    expect(value?.textContent).toBe(
      `${(1540).toLocaleString()}without 9 older runs`,
    );
  });

  it("shows a dash when every run predates lines reviewed", async () => {
    serve({
      metrics: {
        ...metrics,
        totals: {
          ...metrics.totals,
          lines_added: 0,
          lines_removed: 0,
          lines_uncounted_runs: 12,
        },
      },
    });
    render(<MetricsPage />);
    const value = (await screen.findByText("Lines reviewed"))
      .nextElementSibling;
    expect(value?.textContent).toBe("—");
  });

  // The trend averages the six lead days the API sends before the window,
  // so it starts on the window's first day; without them a 3-day window
  // would draw no trend at all. Cost trends its total; throughput trends
  // reviewed runs only, so the skips and the failure move nothing, and
  // its legend says so.
  it("averages the lead days into the trend", async () => {
    serve({
      metrics: {
        ...metrics,
        since: "2026-09-01T00:00:00Z",
        until: "2026-09-04T00:00:00Z",
        lead_since: "2026-08-26T00:00:00Z",
        window: "rolling",
        daily_attempts: [
          { day: "2026-08-30", outcome: "ok", count: 14 },
          { day: "2026-08-30", outcome: "skipped", count: 70 },
          { day: "2026-09-02", outcome: "ok", count: 7 },
          { day: "2026-09-02", outcome: "timed_out", count: 7 },
        ],
        daily: [
          { ...metrics.daily[0], day: "2026-08-28", cost_usd: "0.350" },
          { ...metrics.daily[0], day: "2026-09-02", cost_usd: "0.350" },
        ],
      },
    });
    render(<MetricsPage />);

    // Sep 3: (14 + 7) / 7 reviewed runs, ($0.35 + $0.35) / 7 spend.
    const runs = await screen.findByLabelText(/Runs per day/);
    expect(within(runs as HTMLElement).getByText("3")).toBeDefined();
    const cost = screen.getByLabelText(/Cost per day/);
    expect(within(cost as HTMLElement).getByText("$0.10")).toBeDefined();
    expect(screen.getAllByText("7-day avg")).toHaveLength(1);
    expect(screen.getByText("7-day avg, reviewed only")).toBeDefined();
  });

  // A busy lead week can lift the trend above every bar in the window; the
  // axis grows with it rather than clipping the line.
  it("fits the runs axis to the trend", async () => {
    serve({
      metrics: {
        ...metrics,
        since: "2026-09-01T00:00:00Z",
        until: "2026-09-04T00:00:00Z",
        lead_since: "2026-08-26T00:00:00Z",
        window: "rolling",
        daily_attempts: [
          { day: "2026-08-30", outcome: "ok", count: 700 },
          { day: "2026-09-02", outcome: "ok", count: 7 },
        ],
      },
    });
    render(<MetricsPage />);

    const runs = await screen.findByLabelText(/Runs per day/);
    // (700 + 7) / 7 = 101 on Sep 3, over bars of 7: the top tick is 150,
    // not the 10 the bars alone would give.
    expect(within(runs as HTMLElement).getByText("101")).toBeDefined();
    expect(within(runs as HTMLElement).getByText("150")).toBeDefined();
  });

  it("switches the window to 90 days and back", async () => {
    window.history.replaceState({}, "", "/metrics");
    serve({ metrics });
    render(<MetricsPage />);

    const ninety = await screen.findByRole("button", { name: "90 days" });
    expect(
      screen
        .getByRole("button", { name: "30 days" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    fireEvent.click(ninety);

    await waitFor(() =>
      expect(
        vi
          .mocked(fetch)
          .mock.calls.some(([u]) => /\/metrics\?days=90$/.test(String(u))),
      ).toBe(true),
    );
    expect(window.location.search).toBe("?days=90");
    expect(ninety.getAttribute("aria-pressed")).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "30 days" }));
    expect(window.location.search).toBe("");
  });

  it("opens on the window the URL names", async () => {
    window.history.replaceState({}, "", "/metrics?days=90");
    serve({ metrics });
    render(<MetricsPage />);

    await screen.findByRole("button", { name: "90 days" });
    const calls = vi.mocked(fetch).mock.calls.map(([u]) => String(u));
    expect(calls.some((u) => /\/metrics\?days=90$/.test(u))).toBe(true);
    window.history.replaceState({}, "", "/");
  });

  // 7 days on request (user, 2026-11-10); 30 stays the bare path.
  it("offers 7 days, read from and written to the URL", async () => {
    window.history.replaceState({}, "", "/metrics?days=7");
    serve({ metrics });
    render(<MetricsPage />);

    const seven = await screen.findByRole("button", { name: "7 days" });
    expect(seven.getAttribute("aria-pressed")).toBe("true");
    const calls = vi.mocked(fetch).mock.calls.map(([u]) => String(u));
    expect(calls.some((u) => /\/metrics\?days=7$/.test(u))).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "30 days" }));
    expect(window.location.search).toBe("");
    fireEvent.click(screen.getByRole("button", { name: "7 days" }));
    expect(window.location.search).toBe("?days=7");
    window.history.replaceState({}, "", "/");
  });

  it("reads a window it does not offer as 30 days", async () => {
    window.history.replaceState({}, "", "/metrics?days=14");
    serve({ metrics });
    render(<MetricsPage />);

    await screen.findByRole("button", { name: "30 days" });
    const calls = vi.mocked(fetch).mock.calls.map(([u]) => String(u));
    expect(calls.some((u) => /\/metrics\?days=30$/.test(u))).toBe(true);
    // The address bar says what the page shows, not the window it refused.
    expect(window.location.search).toBe("");
    window.history.replaceState({}, "", "/");
  });

  it("leaves other URL parameters alone when it fixes the window", async () => {
    window.history.replaceState({}, "", "/metrics?days=14&x=1");
    serve({ metrics });
    render(<MetricsPage />);

    await screen.findByRole("button", { name: "30 days" });
    expect(window.location.search).toBe("?x=1");
    fireEvent.click(screen.getByRole("button", { name: "7 days" }));
    expect(window.location.search).toBe("?x=1&days=7");
    window.history.replaceState({}, "", "/");
  });

  it("dashes reviewed when no PR was seen", async () => {
    serve({ metrics: { ...metrics, prs: { reviewed: 0, seen: 0 } } });
    render(<MetricsPage />);

    const value = (await screen.findByText("Reviewed")).nextElementSibling;
    expect(value?.textContent).toBe("—");
  });

  it("labels both charts for a screen reader", async () => {
    serve({ metrics });
    render(<MetricsPage />);

    expect(await screen.findByLabelText(/Cost per day/)).toBeDefined();
    expect(await screen.findByLabelText(/Runs per day/)).toBeDefined();
  });

  // The page asks for the last 30 days, not the calendar month (user's
  // call), and titles itself with the window the server answered.
  it("asks for the last 30 days", async () => {
    serve({
      metrics: {
        ...metrics,
        since: "2026-09-10T00:00:00Z",
        until: "2026-10-10T00:00:00Z",
        window: "rolling",
      },
    });
    render(<MetricsPage />);

    expect(await screen.findByText("the last 30 days")).toBeDefined();
    const calls = vi.mocked(fetch).mock.calls.map(([u]) => String(u));
    expect(calls.some((u) => /\/metrics\?days=30$/.test(u))).toBe(true);
  });

  // Findings and Unpriced get a trend like Cost and Reviewed. Unpriced
  // keeps the flat-line rule: zero all window long draws nothing. The gate
  // reads the daily buckets it draws, not the total (3 here).
  it("draws findings and unpriced trends only when there is one", async () => {
    serve({ metrics });
    render(<MetricsPage />);

    expect(await screen.findByLabelText("Findings trend")).toBeDefined();
    expect(screen.queryByLabelText("Unpriced trend")).toBeNull();
  });

  it("draws the unpriced trend from the daily buckets", async () => {
    serve({
      metrics: {
        ...metrics,
        daily: [{ ...metrics.daily[0], unpriced_runs: 3 }],
      },
    });
    render(<MetricsPage />);

    expect(await screen.findByLabelText("Unpriced trend")).toBeDefined();
  });

  it("says a window is empty rather than drawing an empty chart", async () => {
    serve({
      metrics: {
        ...metrics,
        totals: { ...metrics.totals, cost_usd: null, runs: 0 },
        daily: [],
        daily_attempts: [],
      },
    });
    render(<MetricsPage />);

    await waitFor(() =>
      expect(screen.getByText(/No priced runs/)).toBeDefined(),
    );
    expect(screen.getByText(/No runs in this window/)).toBeDefined();
  });
});

describe("TeamsPage", () => {
  // The page reads its team and its search off the URL, so every test
  // starts on the bare page whatever the one before left behind.
  beforeEach(() => {
    window.history.replaceState({}, "", "/teams");
  });

  /** fullText matches an element whose whole text is `text`, across the
   *  <mark> a search hit splits it into. */
  const fullText = (text: string) => (_: string, el: Element | null) =>
    el?.textContent === text &&
    !Array.from(el.children).some((c) => c.textContent === text);

  const teams: Team[] = [
    {
      slug: "payments",
      // Deliberately out of slug order against "mobile" below, so the page's
      // own sort is what puts them right.
      name: "Zahlungen",
      enabled: true,
      state: "ready",
      repos: -1,
      prs: 9,
      last_reviewed: new Date().toISOString(),
      last_run: new Date().toISOString(),
      claims: [{ project: "PAY" }, { project: "SHARED", repo: "billing-lib" }],
      auto_review_authors: 3,
      ignore_authors: 1,
      exclude_repos: ["*-infra"],
      exclude_source_branches: ["dependabot/*", "wip/*"],
      exclude_target_branches: [],
      model: "gpt-5.5-high",
      model_source: "default",
      riptide: "failing",
      almanach: "ok",
      bitbucket_account: {
        source: "team",
        username: "pay-bot",
        status: "rejected",
        since: new Date().toISOString(),
      },
      jira_account: {
        source: "team",
        status: "ok",
        since: new Date().toISOString(),
      },
    },
    {
      slug: "mobile",
      name: "Mobile",
      enabled: false,
      state: "disabled",
      repos: 0,
      prs: 0,
      last_reviewed: null,
      last_run: null,
      claims: [],
      auto_review_authors: 0,
      ignore_authors: 0,
      exclude_repos: [],
      exclude_source_branches: [],
      exclude_target_branches: [],
      bitbucket_account: null,
      jira_account: null,
    },
  ];

  // A team on its own bot or Jira token is tagged, a broken one says so in
  // the server's fixed word. Not the bot's name: whether the team runs on its
  // own account is what matters (user, 2026-10-26). The same pills sit in
  // the collapsed summary and in the open list, hence two of each.
  it("tags a team's own accounts and their health", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const bots = await screen.findAllByText("Own bot: rejected");
    expect(bots).toHaveLength(2);
    expect(bots[0].getAttribute("title")).toMatch(
      /PUT \/teams\/payments\/credentials/,
    );
    expect(screen.getAllByText("Own Jira token")).toHaveLength(2);
    expect(screen.getAllByText("Riptide: failing")).toHaveLength(2);
    expect(screen.getAllByTitle(/Almanach took the last call/)).toHaveLength(2);
    expect(screen.queryByText(/pay-bot/)).toBeNull();
  });

  // Not set up is a state, shown as one: a team on the shared accounts
  // without riptide says so instead of showing nothing.
  it("shows shared accounts and riptide off", async () => {
    const shared: Team = {
      ...teams[0],
      slug: "hotspot",
      name: "Hotspot",
      riptide: "off",
      almanach: "off",
      model_source: "team",
      bitbucket_account: {
        source: "shared",
        username: "noergler",
        status: null,
        since: null,
      },
      jira_account: { source: "shared", status: null, since: null },
    };
    serve({ teams: { teams: [shared] } });
    render(<TeamsPage />);

    expect(await screen.findAllByText("Shared bot")).toHaveLength(2);
    expect(screen.getAllByText("Shared Jira token")).toHaveLength(2);
    expect(screen.getAllByText("Riptide off")).toHaveLength(2);
    expect(screen.getAllByText("Almanach off")).toHaveLength(2);
    expect(screen.getByText("Team")).toBeDefined();
    expect(screen.queryByText("noergler")).toBeNull();
  });

  // A token riptide refuses after startup is a fault, not an outage: red,
  // not ochre.
  it("shows a refused riptide token as rejected", async () => {
    serve({ teams: { teams: [{ ...teams[0], riptide: "rejected" }] } });
    render(<TeamsPage />);

    const pills = await screen.findAllByText("Riptide: rejected");
    expect(pills).toHaveLength(2);
    expect(pills[0].className).toMatch(/danger/);
  });

  // Almanach is a forwarder like riptide and shows with the same words and
  // tones; a team the server reports nothing for is off.
  it("shows the almanach feed beside riptide", async () => {
    serve({
      teams: {
        teams: [
          { ...teams[0], almanach: "rejected" },
          { ...teams[0], slug: "data", name: "Data", almanach: undefined },
        ],
      },
    });
    render(<TeamsPage />);

    const pills = await screen.findAllByText("Almanach: rejected");
    expect(pills).toHaveLength(2);
    expect(pills[0].className).toMatch(/danger/);
    expect(pills[0].getAttribute("title")).toMatch(/token/);
    expect(screen.getAllByText("Almanach off")).toHaveLength(2);
  });

  // Notes Almanach took and dropped show beside the feed's word, which
  // stays ok for them; none, no pill.
  it("counts the notes Almanach ignored", async () => {
    serve({
      teams: {
        teams: [
          { ...teams[0], almanach: "ok", almanach_ignored_7d: 3 },
          { ...teams[0], slug: "data", name: "Data", almanach: "ok" },
        ],
      },
    });
    render(<TeamsPage />);

    const pills = await screen.findAllByText("3 ignored (7 d)");
    expect(pills).toHaveLength(2);
    expect(pills[0].getAttribute("title")).toMatch(/source repositories/);
    expect(screen.getAllByText(/ignored \(7 d\)/)).toHaveLength(2);
  });

  // Collapsed by default: the list is there to open, the summary pills are
  // what a passing reader sees.
  it("collapses the configuration", async () => {
    serve({ teams: { teams } });
    const { container } = render(<TeamsPage />);

    await screen.findByRole("heading", { name: "Zahlungen" });
    const details = container.querySelectorAll("details");
    expect(details).toHaveLength(1);
    expect(details[0].open).toBe(false);
  });

  it("distinguishes a whole-project claim from a repo one", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    expect(await screen.findByText("PAY")).toBeDefined();
    expect(await screen.findByText("project")).toBeDefined();
    expect(await screen.findByText("SHARED/billing-lib")).toBeDefined();
  });

  // Scope lives in the claims table only. The title line repeated it, and
  // for a whole-project claim a count would be a guess: noergler never learns
  // how many repos a project holds.
  it("keeps scope out of the title line", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    // Once: the PAY row. Not a second time in the title line.
    expect(await screen.findAllByText("project")).toHaveLength(1);
    expect(screen.queryByText(/not started/)).toBeNull();
  });

  // Name and slug side by side read as the same team twice. The slug stays
  // on hover: the disabled note sends the reader to its team= log line.
  it("names a team by its display name, the slug on hover", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const name = await screen.findByRole("heading", { name: "Zahlungen" });
    expect(name.getAttribute("title")).toBe("payments");
    expect(screen.queryByText(/\bpayments\b/)).toBeNull();
  });

  // The rail lists every team under its state, so the roster is one glance
  // and one click; the card count stays the rail's, not a second copy.
  it("indexes the teams by state in the rail", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const rail = await screen.findByRole("navigation", { name: "Teams" });
    const rows = within(rail).getAllByRole("link");
    // The name alone: the PR count (9) is the card's, never the rail's.
    expect(rows.map((r) => r.textContent)).toEqual(["Zahlungen", "Mobile"]);
    expect(rows[0].getAttribute("href")).toBe("/teams/payments");
    expect(within(rail).getByText("Ready")).toBeDefined();
    expect(within(rail).getByText("Disabled")).toBeDefined();
  });

  // The search is over everything a reader might know a team by: its name,
  // its slug, a claim, or a repo it excludes. The miss is dimmed in the
  // rail, not dropped, and its card goes; the hit says what matched.
  it("filters by claim and names the hit in the rail", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const box = await screen.findByRole("searchbox");
    fireEvent.change(box, { target: { value: "billing" } });

    expect(screen.getAllByRole("heading", { level: 3 })).toHaveLength(1);
    expect(screen.getByText(fullText("SHARED/billing-lib"))).toBeDefined();
    const rail = screen.getByRole("navigation", { name: "Teams" });
    expect(
      within(rail).getByText(fullText("↳ SHARED/billing-lib")),
    ).toBeDefined();
    expect(within(rail).getByText("1 of 2 teams match")).toBeDefined();
    expect(window.location.search).toBe("?q=billing");
  });

  // Case and stray spaces are the typist's, not the match's: the rail and
  // the cards read the same normalised query, so neither side disagrees
  // with the other about what hit.
  it("matches regardless of case and whitespace, rail and cards alike", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const box = await screen.findByRole("searchbox");
    fireEvent.change(box, { target: { value: " Billing " } });

    expect(screen.getAllByRole("heading", { level: 3 })).toHaveLength(1);
    const rail = screen.getByRole("navigation", { name: "Teams" });
    expect(within(rail).getByText("1 of 2 teams match")).toBeDefined();
    expect(within(rail).getByText("1/1")).toBeDefined();
    expect(
      within(rail).getByText(fullText("↳ SHARED/billing-lib")),
    ).toBeDefined();
  });

  // A typed query is text, never markup, in the no-match line.
  it("shows a query with angle brackets as typed", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const box = await screen.findByRole("searchbox");
    fireEvent.change(box, { target: { value: "<i>x</i>" } });

    expect(screen.getByText("<i>x</i>")).toBeDefined();
  });

  // A hit in the excluded repos sits behind the collapsed configuration,
  // so the card opens it: a hit nobody can see is not a hit.
  it("opens the configuration for a hit in the excluded repos", async () => {
    serve({ teams: { teams } });
    const { container } = render(<TeamsPage />);

    const box = await screen.findByRole("searchbox");
    fireEvent.change(box, { target: { value: "infra" } });

    expect(screen.getAllByRole("heading", { level: 3 })).toHaveLength(1);
    expect(container.querySelector("details")?.open).toBe(true);
    expect(screen.getByText(fullText("↳ excluded: *-infra"))).toBeDefined();
  });

  it("says when nothing matches and offers the whole list back", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const box = await screen.findByRole("searchbox");
    fireEvent.change(box, { target: { value: "nope" } });

    expect(screen.queryAllByRole("heading", { level: 3 })).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Show all" }));
    expect(screen.getAllByRole("heading", { level: 3 })).toHaveLength(2);
    expect(window.location.search).toBe("");
  });

  // /teams/{slug} is the page scrolled to that team, its configuration
  // open and its rail row current; the # on the card copies that link.
  it("opens the team a link names and copies the link back", async () => {
    window.history.pushState({}, "", "/teams/payments");
    const writeText = vi.fn(async () => {});
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    serve({ teams: { teams } });
    const { container } = render(<TeamsPage />);

    await screen.findByRole("heading", { name: "Zahlungen" });
    expect(container.querySelector("#team-payments")).not.toBeNull();
    expect(container.querySelector("details")?.open).toBe(true);
    const rail = screen.getByRole("navigation", { name: "Teams" });
    expect(
      within(rail)
        .getByRole("link", { name: /Zahlungen/ })
        .getAttribute("aria-current"),
    ).toBe("true");

    fireEvent.click(
      screen.getByRole("button", { name: /Copy link to Zahlungen/ }),
    );
    expect(writeText).toHaveBeenCalledWith(
      `${window.location.origin}/teams/payments`,
    );
    // Said once the clipboard took it, not before.
    expect(await screen.findByText("copied")).toBeDefined();
  });

  // No clipboard API (a plain-http host): the glyph falls back to the
  // selection copy and still only says copied when that worked.
  it("falls back when the clipboard API is missing", async () => {
    vi.stubGlobal("navigator", { ...navigator, clipboard: undefined });
    document.execCommand = vi.fn(() => true);
    serve({ teams: { teams } });
    render(<TeamsPage />);

    await screen.findByRole("heading", { name: "Zahlungen" });
    fireEvent.click(
      screen.getByRole("button", { name: /Copy link to Zahlungen/ }),
    );
    expect(document.execCommand).toHaveBeenCalledWith("copy");
    expect(screen.getByText("copied")).toBeDefined();
  });

  // A stale link says so rather than scrolling nowhere in silence.
  it("reports a slug no team has", async () => {
    window.history.pushState({}, "", "/teams/gone");
    serve({ teams: { teams } });
    render(<TeamsPage />);

    expect(await screen.findByText(/No team/)).toBeDefined();
    expect(screen.getByText("gone")).toBeDefined();
  });

  const rigi: Team = { ...teams[1], slug: "rigi", name: "Rigi" };
  const stanserhorn: Team = {
    ...teams[0],
    slug: "stanserhorn",
    name: "Stanserhorn",
  };

  // Cards in the rail's order, grouped by state then by name: alphabetical
  // cards under a grouped rail put neighbours from different groups side by
  // side (user, 2026-11-10).
  it("orders the cards like the rail", async () => {
    serve({ teams: { teams: [...teams, rigi, stanserhorn] } });
    render(<TeamsPage />);

    await screen.findByRole("heading", { name: "Rigi" });
    const cards = screen
      .getAllByRole("heading", { level: 3 })
      .map((h) => h.textContent);
    const rail = within(screen.getByRole("navigation", { name: "Teams" }))
      .getAllByRole("link")
      .map((a) => a.textContent);
    expect(cards).toEqual(["Stanserhorn", "Zahlungen", "Mobile", "Rigi"]);
    expect(rail.map((r) => r?.trim())).toEqual(cards);
  });

  /** layout fakes what jsdom cannot: an 800px viewport over 2000px of
   *  page (scrollTop 1200 is the end), each card 300px tall at the top
   *  given. Cards top to bottom: Stanserhorn, Zahlungen, Mobile, Rigi. */
  async function layout(path: string) {
    window.history.pushState({}, "", path);
    serve({ teams: { teams: [...teams, rigi, stanserhorn] } });
    const { container } = render(<TeamsPage />);
    await screen.findByRole("heading", { name: "Rigi" });
    const tops: Record<string, number> = {};
    const rect = vi
      .spyOn(Element.prototype, "getBoundingClientRect")
      .mockImplementation(function (this: Element) {
        const card = this.id.startsWith("team-");
        const top = card ? (tops[this.id] ?? 2000) : 0;
        const height = card ? 300 : 800;
        return {
          top,
          bottom: top + height,
          left: 0,
          right: 0,
          width: 0,
          height,
          x: 0,
          y: top,
          toJSON: () => ({}),
        } as DOMRect;
      });
    const scroller = container.firstElementChild as HTMLElement;
    let scrollTop = 0;
    Object.defineProperty(scroller, "scrollTop", { get: () => scrollTop });
    Object.defineProperty(scroller, "clientHeight", { get: () => 800 });
    Object.defineProperty(scroller, "scrollHeight", { get: () => 2000 });
    const rail = within(screen.getByRole("navigation", { name: "Teams" }));
    return {
      /** scrollTo moves the page and fires its scroll event. */
      scrollTo(y: number, s: number, z: number, m: number, r: number) {
        scrollTop = y;
        tops["team-stanserhorn"] = s;
        tops["team-payments"] = z;
        tops["team-mobile"] = m;
        tops["team-rigi"] = r;
        fireEvent.scroll(scroller);
      },
      marked: () =>
        rail
          .getAllByRole("link")
          .filter((a) => a.getAttribute("aria-current") === "true")
          .map((a) => a.textContent?.trim()),
      restore: () => rect.mockRestore(),
    };
  }

  // The focused card in the last screenful cannot reach the top, so the
  // scroll to it ended with the card above it on the spy's line and the
  // rail marked that one too: clicking Stanserhorn lit up Rigi (user,
  // 2026-11-10). The focus keeps the marker until the reader scrolls.
  it("keeps the rail marker on the focused team at the page's end", async () => {
    const page = await layout("/teams/rigi");

    // The page's own scroll bottoms out: Mobile on the spy's line.
    page.scrollTo(1200, -1000, -600, 40, 400);
    expect(page.marked()).toEqual(["Rigi"]);

    // The reader scrolls up 100px, Rigi still on screen: the card at
    // the top has the marker now.
    fireEvent.wheel(window);
    page.scrollTo(1100, -900, -500, 140, 500);
    expect(page.marked()).toEqual(["Zahlungen"]);
    page.restore();
  });

  // A focus higher up does not come back at the end: once the reader
  // scrolls, the end marks the last card, so the last rail row lights up.
  it("marks the last card when the reader scrolls to the end", async () => {
    const page = await layout("/teams/payments");

    fireEvent.wheel(window);
    page.scrollTo(1200, -1000, -600, 40, 400);
    expect(page.marked()).toEqual(["Rigi"]);
    page.restore();
  });

  // A rail click is a navigation: the path becomes the team's, so the URL
  // bar is a copyable link too.
  it("sets the team's path on a rail click", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const rail = await screen.findByRole("navigation", { name: "Teams" });
    fireEvent.click(within(rail).getByRole("link", { name: /Mobile/ }));
    expect(window.location.pathname).toBe("/teams/mobile");
  });

  // The column lists project keys and project/repo pairs (user, 2026-10-27).
  it("heads the claims column Project / repository", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    expect(
      await screen.findByRole("columnheader", { name: "Project / repository" }),
    ).toBeDefined();
  });

  // Unauthenticated and cross-team, so it reports that authors are
  // configured without handing out the roster of who they are.
  it("counts the author lists rather than naming them", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const label = await screen.findByText("Auto-review authors");
    expect(label.nextElementSibling?.textContent).toBe("3");
    expect(screen.queryByText(/renovate/)).toBeNull();
  });

  // Branch globs are shown whole, like the repo globs; an empty list says
  // none rather than vanishing.
  it("shows the excluded branches, set or not", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    expect(await screen.findByText("dependabot/*, wip/*")).toBeDefined();
    const target = screen.getByText("Excluded target branches");
    expect(target.nextElementSibling?.textContent).toBe("none");
  });

  // An empty auto-review list means every author is reviewed, so it reads
  // "all", not "none".
  it("reads an empty auto-review list as all authors", async () => {
    serve({ teams: { teams: [{ ...teams[0], auto_review_authors: 0 }] } });
    render(<TeamsPage />);

    const label = await screen.findByText("Auto-review authors");
    expect(label.nextElementSibling?.textContent).toBe("all");
  });

  // Only an enabled team has a started client, so only it names a model,
  // tagged with where it comes from.
  it("names the model of a team that has one", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    const model = await screen.findByText("gpt-5.5-high");
    expect(model.parentElement?.textContent).toBe("gpt-5.5-highDefault");
    expect(screen.getAllByText(/-high$/)).toHaveLength(1);
  });

  it("points at the log for a disable reason instead of showing one", async () => {
    serve({ teams: { teams } });
    render(<TeamsPage />);

    // The log line to look at, not the reason itself: a disable reason can
    // name an environment variable and this page is unauthenticated.
    expect(await screen.findByText("team_disabled")).toBeDefined();
    expect(screen.queryByText(/OPENAI_API_KEY|WEBHOOK_SECRET/)).toBeNull();
    // Nothing about a disabled team's configuration is known.
    expect(screen.getAllByText("Configuration")).toHaveLength(1);
  });
});

// Static prose: no endpoint, nothing per-instance. What it must not do is
// drift from the reasons the pipeline actually reports.
describe("FaqPage", () => {
  it("answers the question the runs page raises", async () => {
    render(<FaqPage />);

    expect(screen.getByText("Why wasn't my PR reviewed?")).toBeDefined();
    // Worded as review.SkipReason.Label() words them, so the FAQ and a runs
    // row say the same thing.
    expect(screen.getByText("AGENTS.md missing")).toBeDefined();
    expect(screen.getByText("HEAD unchanged since last review")).toBeDefined();
    expect(screen.getByText("PR cost cap reached")).toBeDefined();
  });

  it("names the opt-out keyword", () => {
    render(<FaqPage />);

    expect(screen.getByText("noergloff")).toBeDefined();
  });

  // The page is for developers, not operators: a reader here cannot set env
  // vars, and naming them would send them to the wrong person.
  it("does not hand out env var names", () => {
    const { container } = render(<FaqPage />);
    expect(container.textContent).not.toMatch(/REVIEW_[A-Z_]+/);
    expect(container.textContent).not.toMatch(/BITBUCKET_[A-Z_]+/);
  });
});
