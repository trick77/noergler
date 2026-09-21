import { describe, expect, it } from "vitest";
import { navigate, pageFromPath, pathFor } from "./routing";

describe("routing", () => {
  it("maps paths to pages", () => {
    expect(pageFromPath("/")).toBe("live");
    expect(pageFromPath("/runs")).toBe("runs");
    expect(pageFromPath("/metrics")).toBe("metrics");
    expect(pageFromPath("/teams")).toBe("teams");
  });

  // The Go handler serves the shell only for the paths in its own allowlist,
  // so anything else never reaches the SPA; if it somehow does, live is the
  // safe landing rather than a blank render.
  it("falls back to live for an unknown path", () => {
    expect(pageFromPath("/nope")).toBe("live");
    expect(pageFromPath("")).toBe("live");
  });

  it("round-trips", () => {
    for (const page of ["live", "runs", "metrics", "teams"] as const) {
      expect(pageFromPath(pathFor(page))).toBe(page);
    }
  });

  // The allowlist in backend/web/embed.go must stay in step with these, or
  // a reload on a client route 404s.
  it("agrees with the Go route allowlist", () => {
    expect(pathFor("live")).toBe("/");
    expect(pathFor("runs")).toBe("/runs");
  });
});

describe("navigate", () => {
  // The back button is the only way out of a page, so a duplicate entry per
  // click would make it take several presses to leave.
  it("does not push a duplicate entry for the current page", () => {
    window.history.pushState({}, "", "/runs");
    const before = window.history.length;

    navigate("runs");

    expect(window.location.pathname).toBe("/runs");
    expect(window.history.length).toBe(before);
  });

  it("pushes when the page changes", () => {
    window.history.pushState({}, "", "/");
    navigate("metrics");
    expect(window.location.pathname).toBe("/metrics");
    expect(pageFromPath(window.location.pathname)).toBe("metrics");
  });

  it("uses / for live rather than /live", () => {
    window.history.pushState({}, "", "/teams");
    navigate("live");
    expect(window.location.pathname).toBe("/");
  });
});
