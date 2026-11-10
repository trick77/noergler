// The one chart renderer, in ../netra's manner: every chart on this page is
// this component with different props, and a sparkline is this component with
// all of the furniture switched off. SIZE and FURNITURE are separate knobs.
//
// No charting library, and the component never invents a hue: colours arrive
// as "var(--color-s1)" strings from the caller, so the palette lives in
// index.css where it was validated and nowhere else.
import { useId } from "react";

export interface Series {
  /** Stable identity (a slug, an outcome). Never the name: two teams may
   *  display the same one. */
  id: string;
  name: string;
  /** A CSS variable string, never a hex literal. */
  color: string;
  values: number[];
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

/** stackTops turns a series list into cumulative bar segments. */
function stackTops(series: Series[]): number[][] {
  const tops: number[][] = [];
  let below: number[] = [];
  for (const s of series) {
    const top = s.values.map((v, i) => (below[i] ?? 0) + v);
    tops.push(top);
    below = top;
  }
  return tops;
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
        {mark === "bar"
          ? series.map((s, si) => {
              const bw = Math.min(26, band * 0.62);
              return (
                <g key={s.id}>
                  {s.values.map((v, i) => {
                    if (!v) return null;
                    const base = si === 0 ? 0 : tops[si - 1][i];
                    const yTop = yAt(base + v);
                    const yBot = yAt(base);
                    // The topmost visible segment gets the rounded end; a
                    // 2px gap keeps two segments from reading as one shape.
                    const isTop = series
                      .slice(si + 1)
                      .every((o) => !o.values[i]);
                    return (
                      <rect
                        key={i}
                        x={xAt(i) - bw / 2}
                        y={yTop}
                        width={bw}
                        height={Math.max(1, yBot - yTop - 2)}
                        rx={isTop ? 4 : 0}
                        fill={s.color}
                        fillOpacity={showTrend ? 0.5 : 0.85}
                      />
                    );
                  })}
                </g>
              );
            })
          : series.map((s, si) => {
              const top = tops[si];
              const up = linePath(top);
              const filled = mark === "area";
              const floor = `${xAt(n - 1)},${yAt(dataMin)} L ${xAt(0)},${yAt(dataMin)}`;
              return (
                <g key={s.id}>
                  {filled && (
                    <path
                      d={`M ${up} L ${floor} Z`}
                      fill={s.color}
                      fillOpacity={0.15}
                    />
                  )}
                  <path
                    d={`M ${up}`}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={showTrend ? 1.75 : 2}
                    strokeOpacity={showTrend ? 0.55 : 1}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                </g>
              );
            })}
        {showTrend && (
          <path
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
        <g>
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
