import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunDetail } from "./api";
import { RunPage } from "./RunPage";
import { onNav, pageFromPath, runTokenFromPath } from "./routing";

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

const detail: RunDetail = {
  id: 7,
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
  pr: {
    attempts: [
      {
        id: 6,
        kind: "auto",
        outcome: "ok",
        elapsed_ms: 94000,
        created_at: "2026-10-18T12:02:00Z",
        findings: 6,
        cost_usd: "0.061",
      },
      {
        id: 7,
        kind: "auto",
        outcome: "ok",
        elapsed_ms: 73100,
        created_at: "2026-10-18T13:48:12Z",
        findings: 1,
        cost_usd: "0.048",
      },
    ],
    cost_usd: "0.109",
    cap_usd: "5.000",
    more_before: false,
    more_after: false,
    calls: [],
    more_calls: false,
    calls_failed: false,
  },
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

describe("routing to a run", () => {
  it("reads the link token off the path", () => {
    expect(pageFromPath(`/runs/${TOK}`)).toBe("run");
    expect(runTokenFromPath(`/runs/${TOK}`)).toBe(TOK);
    expect(runTokenFromPath("/runs")).toBeNull();
    // Mirrors the Go allowlist: a numeric attempt id is no run page.
    expect(pageFromPath("/runs/18233")).toBe("runs");
    expect(runTokenFromPath("/runs/18233")).toBeNull();
    expect(runTokenFromPath(`/runs/${TOK}A`)).toBeNull();
  });
});

describe("RunPage", () => {
  it("leads with the summary and links no other run", async () => {
    serve(`runs/${TOK}`, detail);
    const { container } = render(<RunPage token={TOK} />);

    expect(
      await screen.findByText("Retry captures on gateway timeout"),
    ).toBeDefined();
    // The exact third decimal, not money()'s cents.
    expect(screen.getAllByText("$0.048").length).toBeGreaterThan(0);
    // A run's page is reached only from its summary comment.
    for (const a of container.querySelectorAll("a")) {
      expect(a.getAttribute("href") ?? "").not.toMatch(/^\/runs\/./);
    }
    expect(screen.getByText("Open PR ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482",
    );
  });

  // The headline only: the full text is on the PR, behind the link.
  it("shows a finding's headline and links its comment", async () => {
    window.history.replaceState({}, "", `/runs/${TOK}#findings`);
    serve(`runs/${TOK}`, detail);
    render(<RunPage token={TOK} />);

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
    ).toEqual([
      "Overview",
      "Pull request",
      "Findings",
      "Tokens and cost",
      "Prompt and files",
    ]);
    expect(screen.getByText("retry.go:88 ↗").getAttribute("href")).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482/overview?commentId=555",
    );
    expect(screen.getByText("author replied")).toBeDefined();
  });

  // A lane the gateway did not send reads "not sent", never $0.
  it("keeps absent lanes absent and lanes at six decimals", async () => {
    serve(`runs/${TOK}`, detail);
    render(<RunPage token={TOK} />);
    fireEvent.click(
      await screen.findByRole("tab", { name: /Tokens and cost/ }),
    );

    expect(screen.getByText("$0.003552")).toBeDefined();
    expect(screen.getAllByText("not sent").length).toBeGreaterThan(0);
    expect(screen.queryByText("$0.000000")).toBeNull();
    expect(window.location.hash).toBe("#cost");
  });

  // The review's own record: stages on the timeline, the findings that
  // were not posted with why, and the prompt and files tab.
  it("shows the pipeline record", async () => {
    serve(`runs/${TOK}`, {
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
    render(<RunPage token={TOK} />);

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
    expect(screen.getByText("go.sum").getAttribute("title")).toBe(
      "services/payments/go.sum",
    );
  });

  // Replies and answers have no run page, but they are money the PR total
  // counts: listed under the runs, linked to their thread.
  it("lists the PR's replies and answers", async () => {
    window.history.replaceState({}, "", `/runs/${TOK}#pr`);
    serve(`runs/${TOK}`, {
      ...detail,
      pr: {
        ...detail.pr,
        calls: [
          {
            kind: "reply",
            comment_id: 501,
            model: "gpt-5.5",
            cost_usd: "0.004",
            created_at: "2026-10-18T14:00:00Z",
          },
          {
            kind: "question",
            comment_id: null,
            model: "gpt-5.5",
            cost_usd: null,
            created_at: "2026-10-18T14:05:00Z",
          },
        ],
      },
    });
    render(<RunPage token={TOK} />);

    expect(
      (await screen.findByText("reply to a finding ↗")).getAttribute("href"),
    ).toBe(
      "https://bb.example.com/projects/PAY/repos/checkout/pull-requests/1482/overview?commentId=501",
    );
    expect(screen.getByText("answer to a mention")).toBeDefined();
    expect(screen.getByText("$0.004")).toBeDefined();
  });

  it("says why a skipped run has nothing else", async () => {
    serve(`runs/${TOK}`, {
      ...detail,
      id: 9,
      outcome: "skipped",
      reason: "head_unchanged",
      cost_usd: null,
      run: null,
      call: null,
      findings: [],
    });
    render(<RunPage token={TOK} />);
    expect(
      await screen.findByText("HEAD unchanged since last review"),
    ).toBeDefined();
  });
});
