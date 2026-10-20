import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PRPage as PRPageData } from "./api";
import { PRPage } from "./PRPage";

const TOK = "AAAAAAAAAAAAAAAAAAAAAA";

function serve(body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url !== `/api/dashboard/prs/${TOK}`) {
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

describe("PRPage", () => {
  it("leads with the PR's totals and its findings", async () => {
    serve(pr);
    render(<PRPage token={TOK} />);

    expect(
      await screen.findByText("Retry captures on gateway timeout"),
    ).toBeDefined();
    // The cap's total, in cents; the whole PR's lines, not a run's.
    expect(screen.getByText("$0.21")).toBeDefined();
    expect(screen.getByText("+452 / −100")).toBeDefined();
    expect(screen.getByText("12 files")).toBeDefined();
    expect(screen.getByText(/2m 3s in review, 15\.0s waiting/)).toBeDefined();
    expect(
      screen.getAllByRole("tab").map((tab) => tab.textContent?.split(/\d/)[0]),
    ).toEqual(["Findings", "Runs", "Files", "Cost"]);
    // Open first, the runs linked under the PR's page.
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows[0].textContent).toContain("Test sleeps on the wall clock");
    expect(rows[0].textContent).toContain("open");
    expect(rows[1].textContent).toContain("fixed");
    expect(
      screen.getAllByText("Run 3")[0].closest("a")?.getAttribute("href"),
    ).toBe(`/prs/${TOK}/runs/3`);
    expect(screen.getByText("retry.go:88 ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482/overview?commentId=555",
    );
  });

  // One bar per run on one time scale, waits hatched; the waits are the one
  // figure added up across runs.
  it("lists the runs newest first with their time split", async () => {
    window.history.replaceState({}, "", `/prs/${TOK}#runs`);
    serve(pr);
    render(<PRPage token={TOK} />);

    const first = (await screen.findAllByText(/^Run \d$/))[0];
    expect(first.textContent).toBe("Run 3");
    expect(
      screen.getByLabelText(
        "noergler and Bitbucket 30.0s, waiting 12.0s, AI Gateway call 30.0s",
      ),
    ).toBeDefined();
    expect(screen.getByText("15.0s of it waiting")).toBeDefined();
    // A skip has no bar and no changes.
    const skip = screen.getByTitle("HEAD unchanged since last review");
    expect(skip.closest("tr")?.querySelector("[aria-label]")).toBeNull();
    // Replies and answers: money the total counts, linked to their thread.
    expect(screen.getByText("reply to a finding ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482/overview?commentId=501",
    );
  });

  it("names each file with how every run sent it", async () => {
    serve(pr);
    render(<PRPage token={TOK} />);
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
  // the listed runs beside them, and no start date it does not reach.
  it("does not mix a cut history into whole-PR figures", async () => {
    serve({ ...pr, more_runs: true, runs_total: 40 });
    render(<PRPage token={TOK} />);
    expect(await screen.findByText(/^40 runs/)).toBeDefined();
    expect(screen.getByText(/2m 3s in review$/)).toBeDefined();
    expect(screen.queryByText(/waiting/)).toBeNull();
    expect(screen.queryByText(/→/)).toBeNull();
  });

  // No run read the whole diff: a dash, never zeros or a run's increment.
  it("shows an unknown whole-PR scope as a dash", async () => {
    serve({ ...pr, scope: null });
    render(<PRPage token={TOK} />);
    await screen.findByText("Retry captures on gateway timeout");
    expect(screen.queryByText(/files$/)).toBeNull();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("prices each run and the PR against its cap", async () => {
    serve(pr);
    render(<PRPage token={TOK} />);
    fireEvent.click(await screen.findByRole("tab", { name: /Cost/ }));
    expect(screen.getAllByText("$0.21 of $2.00").length).toBeGreaterThan(0);
    expect(screen.getByText("$0.14")).toBeDefined();
    expect(screen.getByText("Input, uncached")).toBeDefined();
    expect(screen.getByText("64,000")).toBeDefined();
  });
});
