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
  totals: {
    runs: 12,
    prompt_tokens: 1000,
    cached_tokens: 0,
    completion_tokens: 100,
    findings_posted: 5,
    cost_usd: "1.250",
    unpriced_runs: 3,
  },
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
    },
  ],
  daily: [{ day: "2026-09-02", team: "payments", runs: 2, cost_usd: "0.500" }],
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
    serve({ runs: { runs }, metrics });
    render(<RunsPage />);

    expect(await screen.findByText("Timed out")).toBeDefined();
    // A skip's pill names its reason, not the bare outcome.
    expect(await screen.findByText("Unchanged")).toBeDefined();
    expect(screen.getAllByText("OK")).toHaveLength(2);
  });

  it("keeps an unpriced run unpriced", async () => {
    serve({ runs: { runs }, metrics });
    render(<RunsPage />);

    expect(await screen.findByText("$0.22")).toBeDefined();
    expect(await screen.findByText("unpriced")).toBeDefined();
    // The failures have no cost at all, which is a dash, not $0.00.
    expect(screen.queryByText("$0.00")).toBeNull();
  });

  // /metrics can fail while /runs succeeds. Four tiles reading 0 above a
  // full table would assert an absence that is really an unknown.
  it("dashes the counts rather than claiming zero when metrics fail", async () => {
    serve({ runs: { runs } });
    render(<RunsPage />);

    await waitFor(() => expect(screen.getByText("PAY/ledger#1")).toBeDefined());
    expect(screen.queryByText("Counts unavailable.")).toBeDefined();
    // The feed is there; the tiles do not claim a number.
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("names the server's month rather than saying this month", async () => {
    serve({ runs: { runs }, metrics });
    render(<RunsPage />);

    expect(
      await screen.findByText(
        "What noergler did with every PR in September 2026, and the latest 50 runs.",
      ),
    ).toBeDefined();
  });

  // Still loading is not zero either: four 0 tiles and "Nothing was skipped"
  // on a slow /metrics are the same false claim as on a failed one.
  it("dashes the counts while metrics are still loading", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        url.includes("/metrics")
          ? new Promise(() => {})
          : Promise.resolve({
              ok: true,
              status: 200,
              json: async () => ({ runs }),
            }),
      ),
    );
    render(<RunsPage />);

    await waitFor(() => expect(screen.getByText("PAY/ledger#1")).toBeDefined());
    expect(screen.getByText("Reviewed").nextElementSibling?.textContent).toBe(
      "—",
    );
    expect(screen.queryByText("Nothing was skipped.")).toBeNull();
  });

  // The tiles were read once while the feed beneath refreshed every 10s, so
  // the counts drifted from the rows for as long as the page stayed open.
  it("refreshes the counts with the feed", async () => {
    serve({ runs: { runs }, metrics });
    render(<RunsPage />);
    await screen.findByText("PAY/ledger#1");
    const metricsCalls = () =>
      (fetch as unknown as { mock: { calls: string[][] } }).mock.calls.filter(
        ([u]) => u.includes("/metrics"),
      ).length;
    expect(metricsCalls()).toBe(1);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(metricsCalls()).toBe(2);
  });

  // Polled, a /metrics that starts failing kept its first reading on the
  // tiles beside a feed that moved on. Failing is unknown, not the old count.
  it("dashes the counts when a metrics refresh fails", async () => {
    let metricsCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.includes("/metrics") && ++metricsCalls > 1) {
          return { ok: false, status: 500, json: async () => ({}) };
        }
        return {
          ok: true,
          status: 200,
          json: async () => (url.includes("/metrics") ? metrics : { runs }),
        };
      }),
    );
    render(<RunsPage />);
    await waitFor(() =>
      expect(
        screen.getByText("Reviewed").nextElementSibling?.textContent,
      ).not.toBe("—"),
    );

    await vi.advanceTimersByTimeAsync(10_000);
    await waitFor(() =>
      expect(screen.getByText("Reviewed").nextElementSibling?.textContent).toBe(
        "—",
      ),
    );
    expect(screen.getByText("Counts unavailable.")).toBeDefined();
  });

  it("ranks the skip reasons by their label", async () => {
    serve({ runs: { runs }, metrics });
    render(<RunsPage />);
    // Once, in the breakdown: on the row the label is the pill's tooltip.
    expect(
      await screen.findAllByText("HEAD unchanged since last review"),
    ).toHaveLength(1);
  });

  // The long label beside the pill wrapped the Outcome cell into three or
  // four lines. The pill says it in a word and keeps the label on hover, so
  // the row still knows why without sending the reader to the logs.
  it("says why a run was skipped in the pill, with the label on hover", async () => {
    serve({ runs: { runs }, metrics });
    render(<RunsPage />);

    const pill = await screen.findByText("Unchanged");
    expect(pill.getAttribute("title")).toBe("HEAD unchanged since last review");
  });

  it("links a PR tag to Bitbucket", async () => {
    serve({
      runs: { runs, bitbucket_url: "https://bitbucket.example.com" },
      metrics,
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
    serve({ runs: { runs, bitbucket_url: "" }, metrics });
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

  it("labels both charts for a screen reader", async () => {
    serve({ metrics });
    render(<MetricsPage />);

    expect(await screen.findByLabelText(/Cost per day/)).toBeDefined();
    expect(await screen.findByLabelText(/Runs per day/)).toBeDefined();
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
    expect(rows.map((r) => r.textContent)).toEqual(["Zahlungen9", "Mobile"]);
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
    expect(container.querySelector("#payments")).not.toBeNull();
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

  it("names the opt-out keyword and points at the API docs", () => {
    render(<FaqPage />);

    expect(screen.getByText("noergloff")).toBeDefined();
    expect(screen.getByText("/api/docs").getAttribute("href")).toBe(
      "/api/docs",
    );
  });

  // The page is for developers, not operators: a reader here cannot set env
  // vars, and naming them would send them to the wrong person.
  it("does not hand out env var names", () => {
    const { container } = render(<FaqPage />);
    expect(container.textContent).not.toMatch(/REVIEW_[A-Z_]+/);
    expect(container.textContent).not.toMatch(/BITBUCKET_[A-Z_]+/);
  });
});
