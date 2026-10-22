import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunDetail } from "./api";
import { RunPane } from "./RunPane";
import { onNav, pageFromPath, prRouteFromPath } from "./routing";

const TOK = "AAAAAAAAAAAAAAAAAAAAAA";

function serve(path: string, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url !== `/api/dashboard/${path}`) {
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

const RUN = `prs/${TOK}/runs/2`;

const detail: RunDetail = {
  n: 2,
  tag: "PAY/checkout#1482",
  title: "Retry captures on gateway timeout",
  team: "payments",
  team_name: "Payments",
  kind: "auto",
  outcome: "ok",
  elapsed_ms: 73100,
  created_at: "2026-10-18T13:48:12Z",
  cost_usd: "0.048",
  run: {
    incremental: true,
    from_commit: "9f8e7d6aaaa",
    to_commit: "3c41a0bbbbb",
    model: "gpt-5.5-medium",
    prompt_tokens: 41210,
    cached_tokens: 28416,
    completion_tokens: 2830,
    findings_posted: 1,
    lines_added: 312,
    lines_removed: 48,
    files_changed: 9,
  },
  call: {
    profile: "gpt-5.5-medium",
    model: "gpt-5.5-2026-08-06",
    input_tokens: 41210,
    input_uncached_tokens: 12794,
    cache_read_tokens: 28416,
    cache_write_tokens: null,
    output_tokens: 2830,
    reasoning_tokens: 1920,
    cost_usd: "0.048",
    cost_input_usd: "0.015993",
    cost_cache_read_usd: "0.003552",
    cost_cache_write_usd: null,
    cost_output_usd: "0.028300",
    cost_reasoning_usd: "0.019200",
    cost_tool_usage_usd: null,
    key_spend_usd: "812.400",
    headers_ms: 58310,
    total_ms: 58420,
    warnings: [],
  },
  findings: [
    {
      severity: "issue",
      confidence: 92,
      file: "retry.go",
      line: 88,
      headline: "Capture is retried without its idempotency key",
      comment: "`RetryPolicy.run` retries the capture with a new key.",
      suggestion: "Reuse the key from the first attempt.",
      dispute: "The gateway dedupes by order id.",
      comment_id: 555,
      replied: true,
      fixed: false,
    },
  ],
  fixed: [],
  bitbucket_url: "https://bb.example.com",
  detail: null,
};

describe("onNav", () => {
  // A modified click is the browser's (new tab, new window); only a plain
  // one stays in the SPA.
  it("leaves modified clicks to the browser", () => {
    window.history.replaceState({}, "", "/runs");
    const click = (mods: Partial<MouseEvent>) => {
      const e = {
        metaKey: false,
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        button: 0,
        preventDefault: vi.fn(),
        ...mods,
      };
      onNav("/runs/7")(e);
      return e.preventDefault;
    };
    expect(click({ metaKey: true })).not.toHaveBeenCalled();
    expect(window.location.pathname).toBe("/runs");
    expect(click({})).toHaveBeenCalled();
    expect(window.location.pathname).toBe("/runs/7");
  });
});

describe("routing to a PR and its runs", () => {
  it("reads the link token and the run number off the path", () => {
    expect(pageFromPath(`/prs/${TOK}`)).toBe("pr");
    expect(prRouteFromPath(`/prs/${TOK}`)).toEqual({ token: TOK, n: null });
    expect(prRouteFromPath(`/prs/${TOK}/runs/12`)).toEqual({
      token: TOK,
      n: 12,
    });
    // Mirrors the Go allowlist: no other spelling of a run, and an old
    // /runs/{token} link is no page.
    for (const path of [
      "/prs",
      `/prs/${TOK}A`,
      `/prs/${TOK}/runs/0`,
      `/prs/${TOK}/runs/01`,
      `/prs/${TOK}/runs`,
      `/runs/${TOK}`,
    ]) {
      expect(prRouteFromPath(path)).toBeNull();
    }
    expect(pageFromPath(`/runs/${TOK}`)).toBe("runs");
  });
});

describe("RunPane", () => {
  // The pane starts at the run's own line: the PR's title, team and link
  // are the shell's around it.
  it("leads with the run's summary", async () => {
    serve(RUN, detail);
    const { container } = render(<RunPane token={TOK} n={2} />);

    const head = await screen.findByRole("heading", { name: "Run 2" });
    // The run's line: outcome, kind, mode, commits and when.
    expect(head.parentElement?.textContent).toContain("ok");
    expect(head.parentElement?.textContent).toContain("auto");
    expect(head.parentElement?.textContent).toContain("incremental");
    expect(head.parentElement?.textContent).toContain("9f8e7d6 → 3c41a0b");
    // Cents, like every amount on the dashboard.
    expect(screen.getAllByText("$0.05").length).toBeGreaterThan(0);
    expect(screen.queryByText("Retry captures on gateway timeout")).toBeNull();
    for (const a of container.querySelectorAll("a")) {
      expect(a.getAttribute("href") ?? "").not.toMatch(/^\/runs\/./);
    }
  });

  // The headline only: the full text is on the PR, behind the link.
  it("shows a finding's headline and links its comment", async () => {
    window.history.replaceState({}, "", `/${RUN}#findings`);
    serve(RUN, detail);
    render(<RunPane token={TOK} n={2} />);

    expect(
      await screen.findByText("Capture is retried without its idempotency key"),
    ).toBeDefined();
    for (const text of [
      "`RetryPolicy.run` retries the capture with a new key.",
      "Reuse the key from the first attempt.",
      "The gateway dedupes by order id.",
    ]) {
      expect(screen.queryByText(text, { exact: false })).toBeNull();
    }
    expect(
      screen.getAllByRole("tab").map((tab) => tab.textContent?.split(/\d/)[0]),
    ).toEqual(["Overview", "Findings", "Tokens and cost", "Prompt and files"]);
    expect(screen.getByText("retry.go:88 ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482/overview?commentId=555",
    );
    expect(screen.getByText("author replied")).toBeDefined();
  });

  // A lane the gateway did not send reads "not sent", never $0.
  it("keeps absent lanes absent and lanes in cents", async () => {
    serve(RUN, detail);
    render(<RunPane token={TOK} n={2} />);
    fireEvent.click(
      await screen.findByRole("tab", { name: /Tokens and cost/ }),
    );

    const row = (lane: string) =>
      screen.getByText(lane).closest("tr")?.textContent ?? "";
    // 0.003552 rounds to cents like every other amount.
    expect(row("Cache read")).toContain("$0.00");
    expect(row("Cache write")).toContain("not sent");
    expect(row("Cache write")).not.toContain("$");
    expect(window.location.hash).toBe("#cost");
  });

  // The review's own record: stages on the timeline, the findings that
  // were not posted with why, and the prompt and files tab.
  it("shows the pipeline record", async () => {
    serve(RUN, {
      ...detail,
      detail: {
        stages: [
          { name: "diff", ms: 1400 },
          { name: "gateway", ms: 58420 },
          { name: "queued", ms: 3100 },
        ],
        prompt: {
          template: 4520,
          files: 22480,
          cumulative_diff: 9870,
          previously_posted: 2150,
          repo_instructions: 1240,
          ticket: 610,
          pr_description: 340,
          total: 41210,
          budget: 628000,
        },
        files: [
          {
            path: "retry.go",
            sent: "full",
            added: 142,
            removed: 18,
            tokens: 6820,
          },
          {
            path: "services/payments/go.sum",
            sent: "not_reviewable",
            added: 2,
            removed: 0,
            tokens: 0,
          },
        ],
        cross_file: ["CaptureClient"],
        agents_md: { found: true, tokens: 1240, warn_tokens: 4000 },
        ticket: "PAY-2291",
        previously_posted: { shown: 6, total: 8 },
        decisions: [
          {
            file: "config.go",
            line: 203,
            severity: "issue",
            headline: "MaxRetries can go negative",
            decision: "dropped",
            reason: "evidence_not_found",
          },
        ],
        moved: [{ file: "retry.go", from: 85, to: 88 }],
        commit: "3c41a0bbbbbbbbbb",
        pre_existing: [
          {
            file: "client/pool.go",
            line: 42,
            headline: "Pool never closes idle connections",
            explanation: "Idle connections pile up until the process restarts.",
            evidence: ["p.idle = append(p.idle, conn)"],
          },
        ],
        verdict: { model: "request_changes", posted: "comment" },
        http: [{ upstream: "bitbucket", calls: 14 }],
      },
    });
    render(<RunPane token={TOK} n={2} />);

    expect((await screen.findAllByText("AI Gateway call")).length).toBe(2);
    expect(screen.getByText("Waiting for the AI Gateway")).toBeDefined();
    // Captioned with the stages' own sum, not the run's elapsed time.
    expect(screen.getByText("1m 3s")).toBeDefined();
    expect(screen.getByText("PAY-2291")).toBeDefined();
    expect(screen.getByText("1,240 of 4,000 tokens (31%)")).toBeDefined();

    fireEvent.click(screen.getByRole("tab", { name: /Findings/ }));
    expect(screen.getByText("MaxRetries can go negative")).toBeDefined();
    // Not posted, so the page says it: the problem, the quoted code and a
    // link to the line at the reviewed commit.
    expect(screen.getByText("client/pool.go:42 ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/browse/client/pool.go?at=3c41a0bbbbbbbbbb#42",
    );
    expect(
      screen.getByText("Idle connections pile up until the process restarts."),
    ).toBeDefined();
    expect(screen.getByText("p.idle = append(p.idle, conn)")).toBeDefined();
    // A drop reason reads as words, not its code.
    expect(screen.getByText("quoted code that is not there")).toBeDefined();
    expect(screen.getByText("moved from :85")).toBeDefined();
    expect(
      screen.getByText(/Verdict lowered from request_changes to comment/),
    ).toBeDefined();

    fireEvent.click(screen.getByRole("tab", { name: /Prompt and files/ }));
    expect(screen.getByText("Cumulative PR diff")).toBeDefined();
    expect(screen.getByText("not reviewed, binary or generated")).toBeDefined();
    expect(
      screen.getByText("go.sum").closest("td")?.getAttribute("title"),
    ).toBe("services/payments/go.sum");
  });

  it("says why a skipped run has nothing else", async () => {
    serve(RUN, {
      ...detail,
      n: 3,
      outcome: "skipped",
      reason: "head_unchanged",
      cost_usd: null,
      run: null,
      call: null,
      findings: [],
    });
    render(<RunPane token={TOK} n={2} />);
    expect(
      await screen.findByText("HEAD unchanged since last review"),
    ).toBeDefined();
  });
});
