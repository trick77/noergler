import { describe, expect, it } from "vitest";
import { ago, duration, money, outcomeTone, scope, teamStateLabel, teamTone, tokens } from "./format";

describe("money", () => {
  // The whole reason cost travels as a string: an unpriced run is not a free
  // run, and rendering it as $0.000 invents a fact the gateway never gave.
  it("says unpriced rather than zero", () => {
    expect(money(null)).toBe("unpriced");
    expect(money("0.000")).toBe("$0.000");
  });

  // Passed through, never parsed: Number("0.218") back to a string is where
  // the exactness the backend preserved would be lost.
  it("preserves the exact decimal it was given", () => {
    expect(money("0.218")).toBe("$0.218");
    expect(money("31.470")).toBe("$31.470");
    expect(money("0.000000001")).toBe("$0.000000001");
  });
});

describe("duration", () => {
  it("scales the unit to the magnitude", () => {
    expect(duration(null)).toBe("—");
    expect(duration(250)).toBe("250ms");
    expect(duration(1500)).toBe("1.5s");
    expect(duration(72400)).toBe("1m 12s");
  });

  // Rounding the seconds remainder independently of the minutes printed
  // "1m 60s": the minutes floor to 1 while 59.5s rounds up to 60.
  it("never prints sixty seconds", () => {
    expect(duration(119500)).toBe("2m 0s");
    expect(duration(119999)).toBe("2m 0s");
    expect(duration(179500)).toBe("3m 0s");
    for (let ms = 60000; ms < 400000; ms += 137) {
      expect(duration(ms)).not.toMatch(/ 60s$/);
    }
  });
});

describe("ago", () => {
  const now = new Date("2026-09-21T12:00:00Z").getTime();
  const at = (iso: string) => ago(iso, now);

  it("never renders a missing timestamp as a date", () => {
    expect(ago(null, now)).toBe("never");
    expect(at("not a date")).toBe("—");
  });

  it("coarsens as it goes back", () => {
    expect(at("2026-09-21T11:59:57Z")).toBe("just now");
    expect(at("2026-09-21T11:59:20Z")).toBe("40s ago");
    expect(at("2026-09-21T11:30:00Z")).toBe("30m ago");
    expect(at("2026-09-21T09:00:00Z")).toBe("3h ago");
    expect(at("2026-09-18T12:00:00Z")).toBe("3d ago");
  });

  // Clock skew between the pod and the browser must not produce "in 4s".
  it("clamps a future timestamp", () => {
    expect(at("2026-09-21T12:00:30Z")).toBe("just now");
  });
});

describe("outcomeTone", () => {
  // Three tones, not one per outcome: ok, a decision, and a fault. Every
  // unknown outcome a newer backend might add reads as a fault rather than
  // silently as success.
  it("maps every outcome to one of three tones", () => {
    expect(outcomeTone("ok")).toBe("ok");
    expect(outcomeTone("skipped")).toBe("skip");
    expect(outcomeTone("timed_out")).toBe("fail");
    expect(outcomeTone("unparseable")).toBe("fail");
    expect(outcomeTone("too_large")).toBe("fail");
    expect(outcomeTone("error")).toBe("fail");
    expect(outcomeTone("invented_later")).toBe("fail");
  });
});

describe("tokens", () => {
  it("abbreviates", () => {
    expect(tokens(950)).toBe("950");
    expect(tokens(18_400_000)).toBe("18.4M");
    expect(tokens(412_000)).toBe("412.0K");
  });
});

describe("team state", () => {
  // The state this was added for: configured, started, owns nothing. Green
  // would claim it is working and red would claim it is broken; ochre is
  // "your move", because somebody has to claim a repository.
  it("gives a team with no repos its own tone", () => {
    expect(teamTone("ready")).toBe("ok");
    expect(teamTone("no_repos")).toBe("skip");
    expect(teamTone("disabled")).toBe("fail");
  });

  it("reads an unknown state as a fault rather than as success", () => {
    expect(teamTone("invented_later")).toBe("fail");
    expect(teamStateLabel("invented_later")).toBe("invented_later");
  });

  it("labels each state", () => {
    expect(teamStateLabel("ready")).toBe("ready");
    expect(teamStateLabel("no_repos")).toBe("no repos");
    expect(teamStateLabel("disabled")).toBe("disabled");
  });
});

describe("scope", () => {
  // A whole-project claim covers every repo in that project, now and every
  // one added later. Printing a number for it would be a guess that goes
  // stale the moment a repo is added.
  it("keeps a whole-project claim distinct from a count", () => {
    expect(scope(-1)).toBe("whole project");
    expect(scope(0)).toBe("no repos");
    expect(scope(1)).toBe("1 repo");
    expect(scope(9)).toBe("9 repos");
  });
});
