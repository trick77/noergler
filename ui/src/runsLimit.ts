/** RUNS_LIMIT is how many of the latest runs the Runs page shows, counts and
 *  filters: the page is those runs and nothing wider (user, 2026-11-10). Its
 *  own module, not RunsPage's: the FAQ names it too, and a component file
 *  exporting a constant breaks fast refresh. */
export const RUNS_LIMIT = 50;
