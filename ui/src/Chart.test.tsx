import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Chart, Sparkline } from "./Chart";
import { _days, _niceTicks, _windowTitle } from "./MetricsPage";

const s1 = (values: number[]) => [{ name: "a", color: "var(--color-s1)", values }];

function svg(el: HTMLElement) {
  const node = el.querySelector("svg");
  if (!node) throw new Error("no svg rendered");
  return node;
}

describe("Chart", () => {
  it("renders nothing drawable for an empty series rather than throwing", () => {
    const { container } = render(<Chart series={[]} width={100} height={40} label="empty" />);
    expect(svg(container).querySelectorAll("path")).toHaveLength(0);
  });

  // The accessible name is the chart's only text for a screen reader: an
  // unlabelled chart is a blank to anyone not looking at it.
  it("carries its label as the image name", () => {
    const { container } = render(
      <Chart series={s1([1, 2, 3])} width={100} height={40} label="a trend" />,
    );
    expect(svg(container).getAttribute("role")).toBe("img");
    expect(svg(container).getAttribute("aria-label")).toBe("a trend");
  });

  // Bars sit on band centres so the first and last stay inside the plot; a
  // line spans endpoint to endpoint. Drawn from the same scale, the outer
  // bars hang half their width outside the axis.
  it("keeps every bar inside the plot", () => {
    const { container } = render(
      <Chart
        series={s1([5, 5, 5])}
        width={300}
        height={120}
        mark="bar"
        ticks={[0, 5]}
        label="bars"
      />,
    );
    const rects = Array.from(svg(container).querySelectorAll("rect"));
    // The clipPath rect is the plot area; the bars are the rest.
    const plot = rects[0];
    const left = Number(plot.getAttribute("x"));
    const right = left + Number(plot.getAttribute("width"));
    for (const bar of rects.slice(1)) {
      const x = Number(bar.getAttribute("x"));
      const w = Number(bar.getAttribute("width"));
      expect(x).toBeGreaterThanOrEqual(left);
      expect(x + w).toBeLessThanOrEqual(right);
    }
  });

  it("draws one band per series when stacked", () => {
    const { container } = render(
      <Chart
        series={[
          { name: "a", color: "var(--color-s1)", values: [1, 2] },
          { name: "b", color: "var(--color-s2)", values: [1, 1] },
        ]}
        width={200}
        height={100}
        mark="stack"
        ticks={[0, 3]}
        label="stack"
      />,
    );
    // Two fills and two strokes.
    expect(svg(container).querySelectorAll("path")).toHaveLength(4);
  });

  // A colour is always the caller's var() string. A hex literal in chart
  // code is how a palette stops being the one that was validated.
  it("never invents a hue", () => {
    const { container } = render(
      <Chart series={s1([1, 2])} width={100} height={40} mark="area" label="x" />,
    );
    for (const path of Array.from(svg(container).querySelectorAll("path"))) {
      const paint = path.getAttribute("fill") ?? path.getAttribute("stroke") ?? "";
      expect(paint.startsWith("#")).toBe(false);
    }
  });

  it("renders a sparkline with no axis or grid", () => {
    const { container } = render(<Sparkline values={[1, 3, 2]} label="spark" />);
    expect(svg(container).querySelectorAll("text")).toHaveLength(0);
    expect(svg(container).querySelectorAll("line")).toHaveLength(0);
  });
});

describe("niceTicks", () => {
  it("tops out at a round number, never the data's own maximum", () => {
    expect(_niceTicks(0)).toEqual([0, 1]);
    expect(_niceTicks(3.7)).toEqual([0, 2, 4]);
    expect(_niceTicks(175)).toEqual([0, 100, 200]);
  });
});

describe("days", () => {
  // The API returns only days that have rows, so the window is filled here:
  // a missing day is a zero, not a gap the chart closes over.
  it("spans [since, until)", () => {
    const out = _days("2026-09-01T00:00:00+02:00", "2026-10-01T00:00:00+02:00");
    expect(out).toHaveLength(30);
    expect(out[0]).toBe("2026-09-01");
    expect(out[29]).toBe("2026-09-30");
  });

  // A month is 28 to 31 days, so nothing may assume a fixed width.
  it("handles months of every length, February included", () => {
    expect(_days("2026-02-01T00:00:00+02:00", "2026-03-01T00:00:00+02:00")).toHaveLength(28);
    expect(_days("2028-02-01T00:00:00+02:00", "2028-03-01T00:00:00+02:00")).toHaveLength(29);
    expect(_days("2026-07-01T00:00:00+02:00", "2026-08-01T00:00:00+02:00")).toHaveLength(31);
  });

  // The first of the month, mid-month: the window starts on the 1st and
  // stops at today rather than running to a month that has not happened.
  it("is empty when the window has not started", () => {
    expect(_days("2026-09-01T00:00:00+02:00", "2026-09-01T00:00:00+02:00")).toHaveLength(0);
  });
});

// The window is the server's calendar month. Reading it back through the
// browser's clock shifted the whole page by a day for any reader west of
// the server: a +02:00 September rendered as Aug 31 to Sep 29 in New York,
// titled itself "August", and dropped Sep 30 off the chart.
describe("the server's month, whatever the viewer's zone", () => {
  const since = "2026-09-01T00:00:00+02:00";
  const until = "2026-10-01T00:00:00+02:00";

  it("spans the server's days, not the viewer's", () => {
    const w = _days(since, until);
    expect(w[0]).toBe("2026-09-01");
    expect(w[w.length - 1]).toBe("2026-09-30");
    expect(w).toHaveLength(30);
  });

  it("titles itself with the server's month", () => {
    const m = { since, until, window: "month" };
    expect(_windowTitle(m as never)).toBe("September 2026");
  });

  // A UTC server is the other direction, and must not shift either.
  it("handles a UTC server", () => {
    const w = _days("2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z");
    expect(w[0]).toBe("2026-09-01");
    expect(w).toHaveLength(30);
  });

  it("renders an empty window rather than guessing at a bad pair", () => {
    expect(_days("not a date", until)).toEqual([]);
  });
});

describe("windowTitle", () => {
  const base = { totals: {}, by_team: [], daily: [], daily_attempts: [], breakdown: [] };

  it("names the month for the default window", () => {
    const m = {
      ...base,
      since: "2026-09-01T00:00:00+02:00",
      until: "2026-10-01T00:00:00+02:00",
      window: "month",
    };
    expect(_windowTitle(m as never)).toBe("September 2026");
  });

  // The title follows what the API answered with, not what the page assumed
  // it asked for.
  it("names a rolling window by its length", () => {
    const m = {
      ...base,
      since: "2026-09-08T00:00:00+02:00",
      until: "2026-09-22T00:00:00+02:00",
      window: "rolling",
    };
    expect(_windowTitle(m as never)).toBe("the last 14 days");
  });
});
