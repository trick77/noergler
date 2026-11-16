// The one chart renderer, in ../netra's manner: every chart on this page is
// this component with different props, and a sparkline is this component with
// all of the furniture switched off. SIZE and FURNITURE are separate knobs.
//
// No charting library, and the component never invents a hue: colours arrive
// as "var(--color-s1)" strings from the caller, so the palette lives in
// index.css where it was validated and nowhere else.
import { useId, useState } from "react";

export interface Series {
  /** Stable identity (a slug, an outcome). Never the name: two teams may
   *  display the same one. */
  id: string;
  name: string;
  /** A CSS variable string, never a hex literal. */
  color: string;
  values: number[];
  /** Bars only: series sharing a key stack; each key is its own bar,
   *  side by side in the day's slot, in order of first appearance. Two
   *  things that must not add up (tasks beside fix requests) get two keys.
   *  Absent: one stack of every series. */
  stack?: string;
}

export type Mark = "line" | "area" | "bar";

/** A trend drawn over the series: a smoothed line with its latest value
 *  printed at the end. While one is drawn the series step back (bars and
 *  lines dimmed) so the trend reads first (user, 2026-11-10). */
export interface Trend {
  /** One per bucket; null where there is nothing to average yet. */
  values: (number | null)[];
  /** A CSS variable string, like a series colour. */
  color: string;
  /** Formats the end label. */
  format: (v: number) => string;
}

export interface ChartProps {
  series: Series[];
  trend?: Trend;
  width: number;
  height: number;
  mark?: Mark;
  /** Tick values on the y axis. Absent means no axis and no grid. */
  ticks?: number[];
  /** Labels under the x axis, one per bucket; "" skips a slot. */
  xLabels?: string[];
  format?: (v: number) => string;
  label: string;
  pad?: number;
}

const PAD = { l: 52, r: 14, t: 12, b: 22 };
// Room right of the plot for a trend's end label, up to "$1234.56".
const TREND_R = 64;
// Between two stacks of one day: apart enough to read as two bars, close
// enough to read as one day.
const STACK_GAP = 3;
// Draw-in: the whole row of bars starts within STAGGER_MS, however many
// days; each stack segment above the first a SEGMENT_MS beat later.
const STAGGER_MS = 400;
const SEGMENT_MS = 90;

/** monotonePath is a smooth curve through the points that never overshoots
 *  between them (Fritsch-Carlson): a plain spline dipped below zero on a
 *  drop and drew a bump the data never had. */
export function monotonePath(pts: [number, number][]): string {
  const f = (v: number) => v.toFixed(1);
  if (pts.length < 3)
    return pts.map(([x, y], i) => `${i ? "L" : "M"} ${f(x)},${f(y)}`).join(" ");
  const n = pts.length;
  const dx: number[] = [];
  const m: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    dx.push(pts[i + 1][0] - pts[i][0]);
    m.push((pts[i + 1][1] - pts[i][1]) / dx[i]);
  }
  const t: number[] = [m[0]];
  for (let i = 1; i < n - 1; i++)
    t.push(m[i - 1] * m[i] <= 0 ? 0 : (m[i - 1] + m[i]) / 2);
  t.push(m[n - 2]);
  for (let i = 0; i < n - 1; i++) {
    if (m[i] === 0) {
      t[i] = 0;
      t[i + 1] = 0;
      continue;
    }
    const a = t[i] / m[i];
    const b = t[i + 1] / m[i];
    const s = a * a + b * b;
    if (s > 9) {
      const k = 3 / Math.sqrt(s);
      t[i] = k * a * m[i];
      t[i + 1] = k * b * m[i];
    }
  }
  let d = `M ${f(pts[0][0])},${f(pts[0][1])}`;
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3;
    d +=
      ` C ${f(pts[i][0] + h)},${f(pts[i][1] + t[i] * h)}` +
      ` ${f(pts[i + 1][0] - h)},${f(pts[i + 1][1] - t[i + 1] * h)}` +
      ` ${f(pts[i + 1][0])},${f(pts[i + 1][1])}`;
  }
  return d;
}

/** stackTops turns a series list into cumulative bar segments, one stack
 *  per `stack` key: each series' tops sit on the series before it in the
 *  same stack. */
export function stackTops(series: Series[]): number[][] {
  const tops: number[][] = [];
  const below = new Map<string, number[]>();
  for (const s of series) {
    const under = below.get(s.stack ?? "") ?? [];
    const top = s.values.map((v, i) => (under[i] ?? 0) + v);
    tops.push(top);
    below.set(s.stack ?? "", top);
  }
  return tops;
}

/** barLayout sizes the bars of one band: one bar per stack key. One stack
 *  keeps the width it always had; more share a wider slot, never thinner
 *  than 1px (a negative width is not drawn at all). A day's bars centre on
 *  its label, counting only the stacks that have something that day, so a
 *  lone fix stack does not sit off its date beside an empty task slot. */
export function barLayout(series: Series[], band: number) {
  const stacks = [...new Set(series.map((s) => s.stack ?? ""))];
  const k = stacks.length;
  const slotW =
    k === 1 ? Math.min(26, band * 0.62) : Math.min(22 * k, band * 0.86);
  const gap = k === 1 ? 0 : Math.min(STACK_GAP, slotW * 0.1);
  const bw = Math.max(1, (slotW - gap * (k - 1)) / k);
  /** Left edge of stack key's bar on day i, relative to the band centre. */
  const offset = (key: string, i: number) => {
    const shown = stacks.filter((st) =>
      series.some((s) => (s.stack ?? "") === st && s.values[i]),
    );
    const j = Math.max(0, shown.indexOf(key));
    const m = Math.max(1, shown.length);
    return -(m * bw + gap * (m - 1)) / 2 + j * (bw + gap);
  };
  return { bw, offset };
}

export function Chart({
  series,
  trend,
  width,
  height,
  mark = "line",
  ticks,
  xLabels,
  format = String,
  label,
  pad = 2,
}: ChartProps) {
  const clipId = useId();
  // The draw-in stagger is for the bars of the first draw only: a bar that
  // is new on a poll grows at once, never waits out a stagger meant for the
  // whole row while the segments around it have already moved. Remembered
  // per bar, not as a "first render" flag: dropping the delay of a bar
  // still growing would restart its timing mid-way.
  const [firstBars] = useState(
    () =>
      new Set(
        series.flatMap((s) =>
          s.values.flatMap((v, i) => (v ? [`${s.id}:${i}`] : [])),
        ),
      ),
  );
  const bare = ticks === undefined;
  const box = bare
    ? { l: 0, r: 0, t: pad, b: pad }
    : trend
      ? { ...PAD, r: TREND_R }
      : PAD;

  const plotW = width - box.l - box.r;
  const plotH = height - box.t - box.b;
  const n = series[0]?.values.length ?? 0;
  if (n === 0 || plotW <= 0 || plotH <= 0) {
    return (
      <svg
        className="chart"
        width={width}
        height={height}
        role="img"
        aria-label={label}
      />
    );
  }

  const tops = mark === "bar" ? stackTops(series) : series.map((s) => s.values);
  const dataMax = Math.max(...tops.flat(), 0);
  const axisMax = ticks && ticks.length > 0 ? Math.max(...ticks) : dataMax || 1;
  // A sparkline baselines at zero, not at the series' own minimum. Against
  // its own min, a run of zeros at one end reads as a solid filled block
  // while a nonzero day reads as the floor: the shape then says the
  // opposite of the data. Zero is also the honest floor for a count.
  const dataMin = 0;
  const span = axisMax - dataMin || 1;

  // A bar sits on its band's centre so the outermost bars stay inside the
  // plot; a line runs endpoint to endpoint so it spans the full width.
  const band = plotW / n;
  const bars = mark === "bar" ? barLayout(series, band) : null;
  const xAt = (i: number) =>
    mark === "bar"
      ? box.l + band * (i + 0.5)
      : box.l + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (v: number) => box.t + plotH - ((v - dataMin) / span) * plotH;

  const linePath = (vals: number[]) =>
    vals.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" L ");

  const trendPts: [number, number][] = [];
  let trendLast: number | null = null;
  trend?.values.forEach((v, i) => {
    if (v === null || i >= n) return;
    trendPts.push([xAt(i), yAt(v)]);
    trendLast = v;
  });
  const showTrend = trend !== undefined && trendPts.length >= 2;
  const trendEnd = trendPts[trendPts.length - 1];

  // Shading (user, 2026-11-16): each series' own hue fading downward, only
  // opacity, never a new colour. Bars per shape (objectBoundingBox), so
  // every bar runs top to floor. Lines over the plot (userSpaceOnUse), so
  // the shade under a low day is faint and under a high one strong, the
  // same scale for every series; per shape, a flat series was unshaded.
  // Several lines overlap their shades, so those stay fainter: the stroke
  // and the legend carry identity, the shade is ground only.
  const [shadeTop, shadeBottom] =
    mark === "bar"
      ? [1, 0.7]
      : mark === "area"
        ? [0.32, 0.02]
        : series.length > 1
          ? [0.14, 0]
          : [0.22, 0];
  const shadeId = (si: number) => `${clipId}-shade-${si}`;
  // Draw-in timing: bars stagger left to right within STAGGER_MS whatever
  // the window's length, a stack's upper segments a beat later; the trend
  // starts once the bars have, its label once it is drawn.
  const step = Math.min(14, STAGGER_MS / n);
  const barDelay = (id: string, si: number, i: number) =>
    firstBars.has(`${id}:${i}`) ? Math.round(i * step) + si * SEGMENT_MS : 0;
  const trendDelay =
    mark === "bar" ? STAGGER_MS + SEGMENT_MS * (series.length - 1) : 350;

  return (
    <svg
      className="chart"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
    >
      <defs>
        <clipPath id={clipId}>
          <rect x={box.l} y={box.t} width={plotW} height={plotH} />
        </clipPath>
        {series.map((s, si) => (
          <linearGradient
            key={s.id}
            id={shadeId(si)}
            x1="0"
            x2="0"
            {...(mark === "bar"
              ? { y1: "0", y2: "1" }
              : {
                  gradientUnits: "userSpaceOnUse",
                  y1: box.t,
                  y2: box.t + plotH,
                })}
          >
            <stop
              offset="0"
              style={{ stopColor: s.color }}
              stopOpacity={shadeTop}
            />
            <stop
              offset="1"
              style={{ stopColor: s.color }}
              stopOpacity={shadeBottom}
            />
          </linearGradient>
        ))}
      </defs>

      {ticks?.map((t) => {
        const y = yAt(t);
        return (
          <g key={t}>
            <line
              x1={box.l}
              x2={width - box.r}
              y1={y}
              y2={y}
              stroke="var(--color-grid-minor)"
              strokeWidth={1}
            />
            <text
              x={box.l - 8}
              y={y + 3.5}
              textAnchor="end"
              fill="var(--color-axis)"
              fontSize={10.5}
              fontFamily="var(--font-mono)"
            >
              {format(t)}
            </text>
          </g>
        );
      })}

      {!bare && (
        <line
          x1={box.l}
          x2={width - box.r}
          y1={height - box.b}
          y2={height - box.b}
          stroke="var(--color-axis)"
          strokeWidth={1}
        />
      )}

      <g clipPath={`url(#${clipId})`}>
        {bars
          ? series.map((s, si) => {
              const key = s.stack ?? "";
              return (
                <g key={s.id}>
                  {s.values.map((v, i) => {
                    if (!v) return null;
                    const base = tops[si][i] - v;
                    const yTop = yAt(base + v);
                    const yBot = yAt(base);
                    // The topmost visible segment of its stack gets the
                    // rounded end; a 2px gap keeps two segments from
                    // reading as one shape.
                    const isTop = series
                      .slice(si + 1)
                      .every((o) => (o.stack ?? "") !== key || !o.values[i]);
                    return (
                      <rect
                        key={i}
                        className="chart-bar"
                        style={{
                          animationDelay: `${barDelay(s.id, si, i)}ms`,
                        }}
                        x={xAt(i) + bars.offset(key, i)}
                        y={yTop}
                        width={bars.bw}
                        height={Math.max(1, yBot - yTop - 2)}
                        rx={isTop ? 4 : 0}
                        fill={`url(#${shadeId(si)})`}
                        fillOpacity={showTrend ? 0.55 : 0.95}
                      />
                    );
                  })}
                </g>
              );
            })
          : // Every shade first, then every line: a later series' shade
            // must never paint over an earlier series' line.
            [
              ...series.map((s, si) => (
                <path
                  key={`${s.id}-fill`}
                  className="chart-fill"
                  d={`M ${linePath(tops[si])} L ${xAt(n - 1)},${yAt(dataMin)} L ${xAt(0)},${yAt(dataMin)} Z`}
                  fill={`url(#${shadeId(si)})`}
                  fillOpacity={showTrend ? 0.6 : 1}
                />
              )),
              // Drawn left to right on first show: pathLength 1 makes the
              // dash the whole line, whatever its length.
              ...series.map((s, si) => (
                <path
                  key={`${s.id}-line`}
                  className="chart-line chart-stroke"
                  pathLength={1}
                  d={`M ${linePath(tops[si])}`}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={showTrend ? 1.75 : 2}
                  strokeOpacity={showTrend ? 0.55 : 1}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              )),
            ]}
        {showTrend && (
          <path
            className="chart-line"
            style={{ animationDelay: `${trendDelay}ms` }}
            pathLength={1}
            d={monotonePath(trendPts)}
            fill="none"
            stroke={trend.color}
            strokeWidth={2.25}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
      </g>

      {/* Outside the clip: the label sits in the right padding. */}
      {showTrend && trendLast !== null && (
        <g
          className="chart-trend-end"
          style={{ animationDelay: `${trendDelay + 700}ms` }}
        >
          <circle
            cx={trendEnd[0]}
            cy={trendEnd[1]}
            r={4}
            fill={trend.color}
            stroke="var(--color-panel)"
            strokeWidth={2}
          />
          <text
            x={trendEnd[0] + 9}
            y={trendEnd[1] + 4}
            fill={trend.color}
            fontSize={11}
            fontWeight={500}
            fontFamily="var(--font-mono)"
          >
            {trend.format(trendLast)}
          </text>
        </g>
      )}

      {xLabels?.map((t, i) =>
        t === "" ? null : (
          <text
            key={i}
            x={xAt(i)}
            y={height - box.b + 14}
            textAnchor="middle"
            fill="var(--color-axis)"
            fontSize={10.5}
            fontFamily="var(--font-mono)"
          >
            {t}
          </text>
        ),
      )}
    </svg>
  );
}

/** Sparkline is the Chart with every piece of furniture off: no axis, no
 *  grid, no labels. A tile's trend, not a chart to read values off. */
export function Sparkline({
  values,
  color = "var(--color-accent-strong)",
  width = 150,
  height = 26,
  label,
}: {
  values: number[];
  color?: string;
  width?: number;
  height?: number;
  label: string;
}) {
  return (
    <Chart
      series={[{ id: label, name: label, color, values }]}
      width={width}
      height={height}
      mark="area"
      label={label}
    />
  );
}
