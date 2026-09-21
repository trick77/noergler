import { useEffect, useRef, useState } from "react";
import { get } from "./api";

export interface Poll<T> {
  data: T | null;
  failed: boolean;
}

/** usePoll fetches one endpoint and, when intervalMs is given, keeps it
 *  fresh.
 *
 *  A failed refresh keeps the last good data on screen and only flags the
 *  failure: a live panel that blanks itself on one dropped request is less
 *  useful than one showing a reading a few seconds old. The blank state is
 *  reserved for having nothing at all.
 *
 *  Every request is aborted on unmount and before the next one, so a slow
 *  response cannot land after the component is gone or overwrite a newer one. */
export function usePoll<T>(path: string, intervalMs?: number): Poll<T> {
  const [data, setData] = useState<T | null>(null);
  const [failed, setFailed] = useState(false);
  // Held in a ref so the effect does not re-run when data arrives, which
  // would restart the interval on every tick.
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    let controller: AbortController | null = null;

    const run = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const next = await get<T>(path, controller.signal);
        if (!alive.current) return;
        setData(next);
        setFailed(false);
      } catch (err) {
        if ((err as Error).name === "AbortError" || !alive.current) return;
        setFailed(true);
      }
    };

    void run();
    const id = intervalMs ? window.setInterval(run, intervalMs) : undefined;
    return () => {
      alive.current = false;
      controller?.abort();
      if (id) window.clearInterval(id);
    };
  }, [path, intervalMs]);

  return { data, failed };
}

/** useNow ticks a clock so an elapsed display advances without refetching.
 *  ../rongo's 500ms tick, for the same reason: the data has not changed, only
 *  how long ago it was. */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(id);
  }, [ms]);
  return now;
}
