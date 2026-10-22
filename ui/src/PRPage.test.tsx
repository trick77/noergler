import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PRPage as PRPageData, RunDetail } from "./api";
import { PRPage } from "./PRPage";

const TOK = "AAAAAAAAAAAAAAAAAAAAAA";

function serve(bodies: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body = bodies[url.replace("/api/dashboard/", "")];
      if (body === undefined) {
        return { ok: false, status: 404, json: async () => ({}) };
      }
      return { ok: true, status: 200, json: async () => body };
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.history.replaceState({}, "", "/");
});

const facts = {
  model: "gpt-5.5-medium",
  prompt_tokens: 0,
  cached_tokens: 0,
  completion_tokens: 0,
  findings_posted: 0,
  lines_added: 0,
  lines_removed: 0,
  files_changed: 0,
};

const pr: PRPageData = {
  tag: "PAY/checkout#1482",
  title: "Retry captures on gateway timeout",
  team: "payments",
  team_name: "Payments",
  state: "open",
  cost_usd: "0.207",
  cap_usd: "2.000",
  scope: { added: 452, removed: 100, files: 12 },
  prompt_tokens: 88000,
  cached_tokens: 24000,
  completion_tokens: 14000,
  elapsed_ms: 123400,
  runs_total: 3,
  wait_ms: 15000,
  runs: [
    {
      n: 1,
      kind: "auto",
      outcome: "ok",
      elapsed_ms: 72000,
      created_at: "2026-10-18T09:41:00Z",
      cost_usd: "0.143",
      run: {
        ...facts,
        incremental: false,
        findings_posted: 2,
        lines_added: 412,
        lines_removed: 88,
        files_changed: 11,
      },
      fixed: 0,
      time: { work_ms: 30000, wait_ms: 12000, gateway_ms: 30000 },
    },
    {
      n: 2,
      kind: "auto",
      outcome: "skipped",
      reason: "head_unchanged",
      elapsed_ms: 400,
      created_at: "2026-10-19T11:05:00Z",
      cost_usd: null,
      run: null,
      fixed: 0,
      time: null,
    },
    {
      n: 3,
      kind: "mention",
      outcome: "ok",
      elapsed_ms: 51000,
      created_at: "2026-10-20T14:02:00Z",
      cost_usd: "0.061",
      run: {
        ...facts,
        incremental: true,
        findings_posted: 0,
        lines_added: 40,
        lines_removed: 12,
        files_changed: 3,
      },
      fixed: 1,
      time: { work_ms: 14000, wait_ms: 3000, gateway_ms: 25000 },
    },
  ],
  more_runs: false,
  findings: [
    {
      severity: "issue",
      file: "retry.go",
      line: 88,
      headline: "Retry loop never resets the backoff",
      comment_id: 555,
      replied: false,
      fixed: true,
      posted_in: 1,
      fixed_in: 3,
    },
    {
      severity: "suggestion",
      file: "retry_test.go",
      line: 57,
      headline: "Test sleeps on the wall clock",
      comment_id: 556,
      replied: true,
      fixed: false,
      posted_in: 1,
      fixed_in: null,
    },
  ],
  files: [
    {
      path: "services/payments/retry.go",
      runs: [
        { n: 1, sent: "full" },
        { n: 3, sent: "diff" },
      ],
      findings: 1,
    },
  ],
  calls: [
    {
      kind: "reply",
      comment_id: 501,
      model: "gpt-5.5",
      cost_usd: "0.004",
      created_at: "2026-10-19T16:20:00Z",
    },
  ],
  more_calls: false,
  calls_failed: false,
  bitbucket_url: "https://bb.example.com",
};

const run3: RunDetail = {
  n: 3,
  tag: "PAY/checkout#1482",
  title: "Retry captures on gateway timeout",
  team: "payments",
  team_name: "Payments",
  kind: "mention",
  outcome: "ok",
  elapsed_ms: 51000,
  created_at: "2026-10-20T14:02:00Z",
  cost_usd: "0.061",
  run: {
    ...facts,
    incremental: true,
    from_commit: "a1b2c3d4e5f6",
    to_commit: "e4f5a6b7c8d9",
    lines_added: 40,
    lines_removed: 12,
    files_changed: 3,
  },
  call: null,
  findings: [],
  fixed: [],
  bitbucket_url: "https://bb.example.com",
  detail: null,
};

const PR = `prs/${TOK}`;
const rail = () =>
  screen.getByRole("navigation", { name: "This pull request" });
const entry = (name: string) =>
  within(rail()).getByText(name).closest("a") as HTMLAnchorElement;

describe("PRPage", () => {
  it("leads with the PR's totals and its findings", async () => {
    serve({ [PR]: pr });
    render(<PRPage token={TOK} n={null} />);

    expect(
      await screen.findByText("Retry captures on gateway timeout"),
    ).toBeDefined();
    // The cap's total, in cents; the whole PR's lines, not a run's.
    expect(screen.getByText("$0.21")).toBeDefined();
    expect(screen.getByText("+452 / −100")).toBeDefined();
    expect(screen.getByText("12 files")).toBeDefined();
    expect(screen.getByText(/2m 3s in review, 15\.0s waiting/)).toBeDefined();
    // The runs are the rail, not a tab.
    expect(
      screen.getAllByRole("tab").map((tab) => tab.textContent?.split(/\d/)[0]),
    ).toEqual(["Findings", "Files", "Cost"]);
    // Open first, the runs linked under the PR's page.
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0].textContent).toContain("Test sleeps on the wall clock");
    expect(rows[0].textContent).toContain("open");
    expect(rows[1].textContent).toContain("fixed");
    expect(
      within(rows[1]).getByText("Run 3").closest("a")?.getAttribute("href"),
    ).toBe(`/prs/${TOK}/runs/3`);
    expect(screen.getByText("retry.go:88 ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482/overview?commentId=555",
    );
  });

  // Newest first, each a link to its pane; the PR's own view is marked.
  it("lists the runs in the rail, newest first", async () => {
    serve({ [PR]: pr });
    render(<PRPage token={TOK} n={null} />);
    await screen.findByText("Retry captures on gateway timeout");

    expect(
      within(rail())
        .getAllByText(/^Run \d$/)
        .map((e) => e.textContent),
    ).toEqual(["Run 3", "Run 2", "Run 1"]);
    expect(entry("Overview").getAttribute("aria-current")).toBe("page");
    expect(entry("Run 3").getAttribute("href")).toBe(`/prs/${TOK}/runs/3`);
    expect(entry("Run 3").getAttribute("aria-current")).toBeNull();
    // A reviewed run carries its counts; a skip its reason and no counts.
    expect(entry("Run 1").textContent).toContain("2 new");
    expect(entry("Run 1").textContent).toContain("$0.14");
    const skip = within(rail()).getByTitle("HEAD unchanged since last review");
    expect(skip.closest("a")?.textContent).not.toContain("new");
    expect(screen.queryByText(/Older runs/)).toBeNull();
  });

  // A run is a pane in the same shell: the PR's header and the rail stay,
  // the rail marks the run.
  it("shows a run beside the rail", async () => {
    serve({ [PR]: pr, [`${PR}/runs/3`]: run3 });
    render(<PRPage token={TOK} n={3} />);

    expect(await screen.findByRole("heading", { name: "Run 3" })).toBeDefined();
    expect(screen.getByText("Retry captures on gateway timeout")).toBeDefined();
    expect(screen.getByText("Open PR ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482",
    );
    expect(entry("Run 3").getAttribute("aria-current")).toBe("page");
    expect(entry("Overview").getAttribute("aria-current")).toBeNull();
    expect(
      screen.getAllByRole("tab").map((tab) => tab.textContent?.split(/\d/)[0]),
    ).toEqual(["Overview", "Findings", "Tokens and cost", "Prompt and files"]);
  });

  // Replies and answers: no run and no page each, so a view of the PR,
  // selected from the rail; money the total counts, linked to their thread.
  it("lists replies and answers from the rail", async () => {
    serve({ [PR]: pr });
    render(<PRPage token={TOK} n={null} />);
    await screen.findByText("Retry captures on gateway timeout");
    fireEvent.click(entry("Replies and answers"));

    expect(screen.getByText("reply to a finding ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482/overview?commentId=501",
    );
    expect(entry("Replies and answers").getAttribute("aria-current")).toBe(
      "page",
    );
    expect(entry("Overview").getAttribute("aria-current")).toBeNull();
    expect(window.location.hash).toBe("#calls");
    expect(screen.queryByRole("tab")).toBeNull();
  });

  // Back from a run to the PR re-reads the hash: the view is the URL's.
  it("reads the view off the URL when a run gives way to the PR", async () => {
    serve({ [PR]: pr, [`${PR}/runs/3`]: run3 });
    window.history.replaceState({}, "", `/prs/${TOK}/runs/3`);
    const { rerender } = render(<PRPage token={TOK} n={3} />);
    await screen.findByRole("heading", { name: "Run 3" });

    window.history.replaceState({}, "", `/prs/${TOK}#files`);
    rerender(<PRPage token={TOK} n={null} />);
    expect(
      screen.getByRole("tab", { name: /Files/ }).getAttribute("aria-selected"),
    ).toBe("true");
    expect(entry("Overview").getAttribute("aria-current")).toBe("page");
  });

  // A stale #calls link on a PR without replies is the overview, not an
  // empty card beside a rail with nothing selected.
  it("shows the overview for #calls when there is nothing to list", async () => {
    window.history.replaceState({}, "", `/prs/${TOK}#calls`);
    serve({ [PR]: { ...pr, calls: [] } });
    render(<PRPage token={TOK} n={null} />);
    await screen.findByText("Retry captures on gateway timeout");

    expect(
      screen
        .getByRole("tab", { name: /Findings/ })
        .getAttribute("aria-selected"),
    ).toBe("true");
    expect(entry("Overview").getAttribute("aria-current")).toBe("page");
    expect(within(rail()).queryByText("Replies and answers")).toBeNull();
  });

  // A failed read lists the entry so the pane can say so, with no count
  // that would read as "none".
  it("names a failed calls read without a count", async () => {
    serve({ [PR]: { ...pr, calls: [], calls_failed: true } });
    render(<PRPage token={TOK} n={null} />);
    await screen.findByText("Retry captures on gateway timeout");
    expect(entry("Replies and answers").textContent).not.toMatch(/\d/);
    fireEvent.click(entry("Replies and answers"));
    expect(screen.getByText(/could not be read/)).toBeDefined();
  });

  it("names each file with how every run sent it", async () => {
    serve({ [PR]: pr });
    render(<PRPage token={TOK} n={null} />);
    fireEvent.click(await screen.findByRole("tab", { name: /Files/ }));
    // The whole path: two files sharing a name are likely across all runs.
    expect(
      screen.getByText("services/payments/").closest("td")?.textContent,
    ).toBe("services/payments/retry.go");
    expect(screen.getByText("full content")).toBeDefined();
    expect(screen.getByText("diff only")).toBeDefined();
    expect(window.location.hash).toBe("#files");
  });

  // A cut history: the whole-history count and time, no waits read from
  // the listed runs beside them, no start date it does not reach, and the
  // rail says the list is cut.
  it("does not mix a cut history into whole-PR figures", async () => {
    serve({ [PR]: { ...pr, more_runs: true, runs_total: 40 } });
    render(<PRPage token={TOK} n={null} />);
    expect(await screen.findByText(/^40 runs/)).toBeDefined();
    expect(screen.getByText(/2m 3s in review$/)).toBeDefined();
    expect(screen.queryByText(/waiting/)).toBeNull();
    expect(screen.queryByText(/→/)).toBeNull();
    expect(within(rail()).getByText(/Older runs/)).toBeDefined();
  });

  // No run read the whole diff: a dash, never zeros or a run's increment.
  it("shows an unknown whole-PR scope as a dash", async () => {
    serve({ [PR]: { ...pr, scope: null } });
    render(<PRPage token={TOK} n={null} />);
    await screen.findByText("Retry captures on gateway timeout");
    expect(screen.queryByText("12 files")).toBeNull();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("prices each run and the PR against its cap", async () => {
    serve({ [PR]: pr });
    render(<PRPage token={TOK} n={null} />);
    fireEvent.click(await screen.findByRole("tab", { name: /Cost/ }));
    expect(screen.getAllByText("$0.21 of $2.00").length).toBeGreaterThan(0);
    expect(screen.getAllByText("$0.14").length).toBeGreaterThan(0);
    expect(screen.getByText("Input, uncached")).toBeDefined();
    expect(screen.getByText("64,000")).toBeDefined();
  });
});
