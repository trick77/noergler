// The one chart renderer, in ../netra's manner: every chart on this page is
// this component with different props, and a sparkline is this component with
// all of the furniture switched off. SIZE and FURNITURE are separate knobs.
//
// No charting library, and the component never invents a hue: colours arrive
// as "var(--color-s1)" strings from the caller, so the palette lives in
// index.css where it was validated and nowhere else.
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

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

/** A trend drawn over the series: a smoothed line with the value of its
 *  last non-null point printed past the last day's bar or point. That point need not be
 *  the last bucket: the metrics page ends its trend on yesterday. While one
 *  is drawn the series step back (bars and lines dimmed) so the trend reads
 *  first (user, 2026-11-10). */
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
  /** Buckets drawn on a faint band behind the marks: the weekends, so a
   *  Saturday dip explains itself. */
  bands?: boolean[];
  /** The last bucket is still filling (today): its bars are outlines, its
   *  line segment dashed, so a half day never reads as a drop. */
  partialLast?: boolean;
  /** The hovered bucket, shared by every chart on the page: each draws its
   *  crosshair on the same day. */
  hover?: number | null;
  onHover?: (i: number | null) => void;
  /** What the hovered bucket shows in this chart; only the chart under the
   *  pointer passes it. */
  tip?: (i: number) => ReactNode;
  /** A series id to bring forward; the others step back. An id this chart
   *  does not draw changes nothing (a team on the outcome chart). */
  focus?: string | null;
  /** A dot on the last point, for a sparkline. */
  endDot?: boolean;
  /** Keep the trend label's room on the right even without a trend, so a
   *  chart lines its days up with one stacked above or below it. */
  reserveTrend?: boolean;
}

const PAD = { l: 52, r: 14, t: 12, b: 22 };
// Room right of the plot for a trend's end label, up to "$1234.56".
const TREND_R = 64;
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

/** stackTops turns a series list into cumulative bar segments: each
 *  series' tops sit on the series before it. One stack a day: two things
 *  that must not add up go in two charts, never two stacks of one (the
 *  Activity card's switch, user, 2026-11-17). */
export function stackTops(series: Series[]): number[][] {
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
  bands,
  partialLast = false,
  hover = null,
  onHover,
  tip,
  focus = null,
  endDot = false,
  reserveTrend = false,
}: ChartProps) {
  const clipId = useId();
  // The day last reported, and where the svg sits on screen for the tip.
  const lastDay = useRef<number | null>(null);
  const [anchor, setAnchor] = useState<{
    left: number;
    top: number;
    scale: number;
    vw: number;
  } | null>(null);
  // A fixed tip would stay put while the page scrolls under it: any scroll
  // hides it until the pointer moves again.
  useEffect(() => {
    if (!anchor) return;
    const off = () => setAnchor(null);
    window.addEventListener("scroll", off, { capture: true, passive: true });
    return () => window.removeEventListener("scroll", off, { capture: true });
  }, [anchor]);
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
  // The end dot (r 3 plus its 2px ring) needs room past the last point.
  const box = bare
    ? endDot
      ? { l: 0, r: 5, t: Math.max(pad, 5), b: pad }
      : { l: 0, r: 0, t: pad, b: pad }
    : trend || reserveTrend
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
  // plot. A line on an axis does too: a crosshair then falls on the same
  // day in a bar chart and a line chart side by side. Only a sparkline's
  // line runs endpoint to endpoint, to span the tile.
  const band = plotW / n;
  const bw = Math.min(26, band * 0.62);
  const centred = mark === "bar" || !bare;
  const xAt = (i: number) =>
    centred
      ? box.l + band * (i + 0.5)
      : box.l + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (v: number) => box.t + plotH - ((v - dataMin) / span) * plotH;

  const linePath = (vals: number[], from = 0) =>
    vals
      .slice(from)
      .map((v, i) => `${xAt(from + i)},${yAt(v)}`)
      .join(" L ");
  // The still-filling day is its own dashed segment; the solid line stops
  // the day before it.
  const tail = partialLast && n >= 2;
  const solidUpTo = (vals: number[]) => (tail ? vals.slice(0, -1) : vals);

  const focusing = focus !== null && series.some((s) => s.id === focus);
  const dim = (id: string) => focusing && id !== focus;
  const shown = hover !== null && hover >= 0 && hover < n ? hover : null;

  const trendPts: [number, number][] = [];
  let trendLast: number | null = null;
  trend?.values.forEach((v, i) => {
    if (v === null || i >= n) return;
    trendPts.push([xAt(i), yAt(v)]);
    trendLast = v;
  });
  const showTrend = trend !== undefined && trendPts.length >= 2;
  const trendEnd = trendPts[trendPts.length - 1];
  // The label starts past the last day's bar (a line: its point), whichever
  // day the trend ends on, and never closer than 9px to its dot. A trend that stops before
  // today (the metrics page) printed its label over today's bar; one put in
  // the right padding instead floated half a band off a dot on the last day.
  const lastRight = xAt(n - 1) + (mark === "bar" ? bw / 2 : 0);
  const trendLabelX = trendEnd ? Math.max(trendEnd[0] + 9, lastRight + 6) : 0;

  // Shading (user, 2026-11-16): each series' own hue fading downward, only
  // opacity, never a new colour. Bars per shape (objectBoundingBox), so
  // every bar runs top to floor. Lines over the plot (userSpaceOnUse), so
  // the shade under a low day is faint and under a high one strong, the
  // same scale for every series; per shape, a flat series was unshaded.
  // Several lines get no shade at all (user, 2026-11-19): four overlapping shades and a trend read as murk, and the
  // stroke and the legend carry identity anyway.
  const [shadeTop, shadeBottom] =
    mark === "bar" ? [1, 0.7] : mark === "area" ? [0.32, 0.02] : [0.22, 0];
  const shaded = mark !== "line" || series.length === 1;
  const shadeId = (si: number) => `${clipId}-shade-${si}`;
  // Draw-in timing: bars stagger left to right within STAGGER_MS whatever
  // the window's length, a stack's upper segments a beat later; the trend
  // starts once the bars have, its label once it is drawn.
  const step = Math.min(14, STAGGER_MS / n);
  const barDelay = (id: string, si: number, i: number) =>
    firstBars.has(`${id}:${i}`) ? Math.round(i * step) + si * SEGMENT_MS : 0;
  const trendDelay =
    mark === "bar" ? STAGGER_MS + SEGMENT_MS * (series.length - 1) : 350;

  const svg = (
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

      {bands?.map((on, i) =>
        on && i < n ? (
          <rect
            key={`band-${i}`}
            x={box.l + band * i}
            y={box.t}
            width={band}
            height={plotH}
            fill="var(--color-active)"
            fillOpacity={0.5}
          />
        ) : null,
      )}

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
              return (
                <g key={s.id}>
                  {s.values.map((v, i) => {
                    if (!v) return null;
                    const base = tops[si][i] - v;
                    const yTop = yAt(base + v);
                    const yBot = yAt(base);
                    // The topmost visible segment gets the rounded end; a
                    // 2px gap keeps two segments from reading as one shape.
                    const isTop = series
                      .slice(si + 1)
                      .every((o) => !o.values[i]);
                    const h = Math.max(1, yBot - yTop - 2);
                    const opacity = dim(s.id)
                      ? 0.18
                      : i === shown
                        ? 1
                        : showTrend
                          ? 0.55
                          : 0.95;
                    // Today's bar is an outline: the day is not over.
                    if (tail && i === n - 1) {
                      return (
                        <rect
                          key={i}
                          className="chart-bar"
                          style={{
                            animationDelay: `${barDelay(s.id, si, i)}ms`,
                          }}
                          x={xAt(i) - bw / 2 + 0.75}
                          y={yTop + 0.75}
                          width={Math.max(0, bw - 1.5)}
                          height={Math.max(0.5, h - 1.5)}
                          rx={isTop ? 3.5 : 0}
                          fill="none"
                          stroke={s.color}
                          strokeWidth={1.5}
                          strokeOpacity={opacity}
                        />
                      );
                    }
                    return (
                      <rect
                        key={i}
                        className="chart-bar"
                        style={{
                          animationDelay: `${barDelay(s.id, si, i)}ms`,
                        }}
                        x={xAt(i) - bw / 2}
                        y={yTop}
                        width={bw}
                        height={h}
                        rx={isTop ? 4 : 0}
                        fill={`url(#${shadeId(si)})`}
                        fillOpacity={opacity}
                      />
                    );
                  })}
                </g>
              );
            })
          : // Every shade first, then every line: a later series' shade
            // must never paint over an earlier series' line.
            [
              ...(shaded
                ? series.map((s, si) => (
                    <path
                      key={`${s.id}-fill`}
                      className="chart-fill"
                      d={`M ${linePath(tops[si])} L ${xAt(n - 1)},${yAt(dataMin)} L ${xAt(0)},${yAt(dataMin)} Z`}
                      fill={`url(#${shadeId(si)})`}
                      fillOpacity={showTrend ? 0.6 : 1}
                    />
                  ))
                : []),
              // Drawn left to right on first show: pathLength 1 makes the
              // dash the whole line, whatever its length.
              ...series.flatMap((s, si) => {
                const lit = focusing && s.id === focus;
                const strokeOpacity = dim(s.id)
                  ? 0.15
                  : lit
                    ? 1
                    : showTrend
                      ? 0.55
                      : 1;
                const strokeWidth = lit ? 2.5 : showTrend ? 1.75 : 2;
                const solid = solidUpTo(tops[si]);
                const out = [
                  <path
                    key={`${s.id}-line`}
                    className="chart-line chart-stroke"
                    pathLength={1}
                    d={`M ${linePath(solid)}`}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={strokeWidth}
                    strokeOpacity={strokeOpacity}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />,
                ];
                // The draw-in dash would fight this one's, so it fades in
                // instead.
                if (tail) {
                  out.push(
                    <path
                      key={`${s.id}-tail`}
                      className="chart-fill"
                      d={`M ${linePath(tops[si], n - 2)}`}
                      fill="none"
                      stroke={s.color}
                      strokeWidth={strokeWidth}
                      strokeOpacity={strokeOpacity}
                      strokeDasharray="3 4"
                      strokeLinecap="round"
                    />,
                  );
                }
                return out;
              }),
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

      {/* Outside the clip: the label runs on into the right padding. */}
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
            x={trendLabelX}
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
            fill={
              tail && i === n - 1 ? "var(--color-muted)" : "var(--color-axis)"
            }
            fontSize={10.5}
            fontFamily="var(--font-mono)"
          >
            {t}
          </text>
        ),
      )}

      {endDot && (
        <circle
          className="chart-trend-end"
          style={{ animationDelay: "700ms" }}
          cx={xAt(n - 1)}
          cy={yAt(series[0].values[n - 1])}
          r={3}
          fill={tail ? "var(--color-panel)" : series[0].color}
          stroke={tail ? series[0].color : "var(--color-panel)"}
          strokeWidth={tail ? 1.5 : 2}
        />
      )}

      {shown !== null && (
        <line
          x1={xAt(shown)}
          x2={xAt(shown)}
          y1={box.t}
          y2={box.t + plotH}
          stroke="var(--color-muted)"
          strokeOpacity={0.7}
          strokeWidth={1}
          pointerEvents="none"
        />
      )}

      {/* One catcher over the plot, not a rect a day: the pointer's x
          picks the day, scaled back when the svg is drawn narrower. */}
      {onHover && (
        <rect
          x={box.l}
          y={box.t}
          width={plotW}
          height={plotH + box.b}
          fill="transparent"
          onPointerMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const x = ((e.clientX - r.left) / (r.width || plotW)) * plotW;
            const i = Math.max(0, Math.min(n - 1, Math.floor(x / band)));
            // Only a new day is news: every pixel of movement used to
            // re-render the page.
            if (i !== lastDay.current) {
              lastDay.current = i;
              onHover(i);
            }
            const s = (
              e.currentTarget.ownerSVGElement ?? e.currentTarget
            ).getBoundingClientRect();
            const a = {
              left: s.left,
              top: s.top,
              scale: s.width / width || 1,
              vw: document.documentElement.clientWidth,
            };
            setAnchor((p) =>
              p &&
              p.left === a.left &&
              p.top === a.top &&
              p.scale === a.scale &&
              p.vw === a.vw
                ? p
                : a,
            );
          }}
          onPointerLeave={() => {
            lastDay.current = null;
            onHover(null);
          }}
        />
      )}
    </svg>
  );

  if (!onHover) return svg;
  // Fixed to the viewport, not placed in the chart: the card around it
  // clips (overflow-hidden), and a tall tip or one near the card's edge
  // was cut off. Beside the crosshair, on whichever side has the room.
  let tipStyle: CSSProperties | null = null;
  if (anchor && shown !== null) {
    const px = anchor.left + xAt(shown) * anchor.scale;
    tipStyle = {
      top: anchor.top + 4,
      ...(xAt(shown) > width * 0.55
        ? { right: anchor.vw - px + 12 }
        : { left: px + 12 }),
    };
  }
  return (
    <div style={{ maxWidth: width }}>
      {svg}
      {tip && shown !== null && tipStyle && (
        <div
          className="pointer-events-none fixed z-20 min-w-[150px] rounded-ui-sm border border-elevated-border bg-elevated px-2.5 py-2 text-[12px] text-ink-dim shadow-menu"
          style={tipStyle}
        >
          {tip(shown)}
        </div>
      )}
    </div>
  );
}

/** useWidth is the rendered width of the box the ref sits on, followed as
 *  it changes. Before the first measure (and in jsdom, which lays nothing
 *  out) it is the fallback. */
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setWidth(Math.floor(el.getBoundingClientRect().width));
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width > 0 ? width : fallback] as const;
}

/** FluidChart is the Chart drawn at its column's own width rather than at
 *  a fixed one scaled down: in a half-width card a 1290 px chart shrank its
 *  axis text to half size. Labels depend on the width (how many fit), so
 *  they come as a function of it. */
export function FluidChart({
  xLabels,
  ...props
}: Omit<ChartProps, "width" | "xLabels"> & {
  xLabels?: (width: number) => string[];
}) {
  const [ref, width] = useWidth(1290);
  return (
    <div ref={ref} className="min-w-0">
      <Chart {...props} width={width} xLabels={xLabels?.(width)} />
    </div>
  );
}

/** labelEvery is how many buckets apart the x labels of a chart this wide
 *  (drawn with the trend's room, as the metrics charts are) must sit so
 *  that none collides with the next: minPx apiece. */
export function labelEvery(width: number, n: number, minPx = 48): number {
  const plotW = width - PAD.l - TREND_R;
  if (n === 0 || plotW <= 0) return 1;
  return Math.max(1, Math.ceil(minPx / (plotW / n)));
}

/** Sparkline is the Chart with every piece of furniture off: no axis, no
 *  grid, no labels. A tile's trend, not a chart to read values off. It
 *  spans its tile (user, 2026-11-19: a fixed 150 px left most of a wide
 *  tile empty) and marks where it ends. */
export function Sparkline({
  values,
  color = "var(--color-accent-strong)",
  height = 30,
  label,
  partial = false,
}: {
  values: number[];
  color?: string;
  height?: number;
  label: string;
  /** The last value is today's, still filling: a dashed last segment and
   *  a hollow end, as on the charts below. */
  partial?: boolean;
}) {
  const [ref, width] = useWidth(150);
  return (
    <div ref={ref} className="min-w-0">
      <Chart
        series={[{ id: label, name: label, color, values }]}
        width={width}
        height={height}
        mark="area"
        label={label}
        endDot
        partialLast={partial}
      />
    </div>
  );
}
