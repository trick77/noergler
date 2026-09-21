import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePoll } from "./usePoll";

function Probe({ interval }: { interval?: number }) {
  const { data, failed } = usePoll<{ n: number }>("live", interval);
  if (failed) return <p>failed</p>;
  if (!data) return <p>loading</p>;
  return <p>n={data.n}</p>;
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

describe("usePoll", () => {
  // The bug this exists for: every tick used to abort the request in flight,
  // so a response slower than the interval was cancelled a moment before it
  // arrived, every time. The AbortError path returns before setFailed, so
  // the page sat on its skeleton forever and never said anything was wrong.
  it("lets a response slower than the interval land", async () => {
    let resolve!: (v: unknown) => void;
    const slow = new Promise((r) => {
      resolve = r;
    });
    let calls = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: { signal?: AbortSignal }) => {
        calls += 1;
        await slow;
        if (init?.signal?.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
        return { ok: true, status: 200, json: async () => ({ n: calls }) };
      }),
    );

    render(<Probe interval={50} />);
    expect(screen.getByText("loading")).toBeDefined();

    // Several intervals pass while the first request is still outstanding.
    await vi.advanceTimersByTimeAsync(250);
    // Skipped, not stacked: one request is in flight, not six.
    expect(calls).toBe(1);

    resolve(null);
    await waitFor(() => expect(screen.getByText("n=1")).toBeDefined());
  });

  it("keeps polling after a slow response lands", async () => {
    let calls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        calls += 1;
        return { ok: true, status: 200, json: async () => ({ n: calls }) };
      }),
    );

    render(<Probe interval={50} />);
    await waitFor(() => expect(screen.getByText("n=1")).toBeDefined());

    await vi.advanceTimersByTimeAsync(120);
    await waitFor(() => expect(calls).toBeGreaterThan(1));
  });

  // A dropped request says so, rather than showing stale data as current or
  // hanging on the skeleton.
  it("reports a failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 500, json: async () => ({}) })),
    );

    render(<Probe />);
    await waitFor(() => expect(screen.getByText("failed")).toBeDefined());
  });

  // A refresh that fails keeps the last good reading on screen: a live panel
  // showing a value a few seconds old beats one that blanks itself.
  it("keeps the last good data when a refresh fails", async () => {
    let calls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        calls += 1;
        if (calls === 1) return { ok: true, status: 200, json: async () => ({ n: 1 }) };
        return { ok: false, status: 500, json: async () => ({}) };
      }),
    );

    render(<Probe interval={50} />);
    await waitFor(() => expect(screen.getByText("n=1")).toBeDefined());

    await vi.advanceTimersByTimeAsync(120);
    await waitFor(() => expect(calls).toBeGreaterThan(1));
    // Still the reading, not a blank.
    expect(screen.queryByText("loading")).toBeNull();
  });
});
