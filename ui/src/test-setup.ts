// jsdom does not implement these, and the chart and poll code both touch
// them. Stubbed once here rather than in every test file.
if (!window.matchMedia) {
  window.matchMedia = ((q: string) => ({
    matches: false,
    media: q,
    addEventListener() {},
    removeEventListener() {},
  })) as unknown as typeof window.matchMedia;
}
