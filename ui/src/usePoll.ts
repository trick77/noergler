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
    let inFlight = false;

    const run = async () => {
      // A tick while a request is still running is SKIPPED, not a reason to
      // abort and retry. Aborting each time meant a response slower than the
      // interval never landed: the AbortError returns below before
      // setFailed, so the page sat on its loading skeleton forever while
      // every request was cancelled a moment before it arrived. /live does a
      // DB read on every poll, so 3s is well within reach.
      if (inFlight) return;
      inFlight = true;
      controller = new AbortController();
      try {
        const next = await get<T>(path, controller.signal);
        if (!alive.current) return;
        setData(next);
        setFailed(false);
      } catch (err) {
        if ((err as Error).name === "AbortError" || !alive.current) return;
        setFailed(true);
      } finally {
        inFlight = false;
      }
    };

    void run();
    const id = intervalMs ? window.setInterval(run, intervalMs) : undefined;
    return () => {
      // Unmount still aborts: that request's result has nowhere to go.
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
