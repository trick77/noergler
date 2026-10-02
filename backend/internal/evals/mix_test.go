package evals

import (
	"context"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// sampledRun is a three-sample run of the one fixture case, labelled and
// priced: the first sample at cold, the rest at warm.
func sampledRun(model string, cold, warm int64, script ...[]inference.ReviewFinding) Sampled {
	client := &scriptedReviewer{script: script, costs: []*int64{&cold, &warm, &warm}}
	s := RunSampled(context.Background(), client, "{files}", oneCase(seeded), countStub, 3, 1, nil)
	s.Model, s.Effort = model, "some-level"
	return s
}

// A primary that catches the bug once in three, voted with a secondary that
// always catches it: two secondary votes carry every group.
func TestMix_SecondaryVotesCarryAWeakPrimary(t *testing.T) {
	primary := sampledRun("m", 1000, 400, bug(12), nil, nil)
	secondary := sampledRun("m", 800, 400, bug(12), bug(13), bug(12))
	got, err := Mix(oneCase(seeded), primary, secondary)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct{ caught, calls, cost float64 }{
		ViewPrimary: {1.0 / 3, 1, 1},
		ViewMixVote: {1, 3, 1.8},
		// A catching primary agrees with the first secondary and stops at
		// two calls; a missing one needs the third.
		ViewMixAdaptive: {1, (2.0 + 3 + 3) / 3, (1000 + (8.0/3-1)*400) / 1000},
	}
	for name, w := range want {
		v := findView(got.Views, name)
		if !near(v.Caught, w.caught) || !near(v.Calls, w.calls) {
			t.Errorf("%s: caught %.3f in %.3f call(s), want %.3f in %.3f", name, v.Caught, v.Calls, w.caught, w.calls)
		}
		if v.CostMultiple == nil || !near(*v.CostMultiple, w.cost) {
			t.Errorf("%s: cost multiple %v, want %.3f", name, v.CostMultiple, w.cost)
		}
	}
	var b strings.Builder
	got.Report(&b)
	if out := b.String(); !strings.Contains(out, "no calls made") || !strings.Contains(out, ViewMixVote) {
		t.Errorf("report:\n%s", out)
	}
}

// Another model reads no cache the primary wrote, so its first call is cold.
func TestMix_AnotherModelPaysItsOwnColdCall(t *testing.T) {
	primary := sampledRun("big", 1000, 400, nil, nil, nil)
	secondary := sampledRun("small", 500, 100, nil, nil, nil)
	got, err := Mix(oneCase(seeded), primary, secondary)
	if err != nil {
		t.Fatal(err)
	}
	if m := findView(got.Views, ViewMixVote).CostMultiple; m == nil || !near(*m, 1.6) {
		t.Errorf("cost multiple %v, want (1000+500+100)/1000", m)
	}
}

func TestMix_RefusesWhatItCannotScore(t *testing.T) {
	good := sampledRun("m", 1, 1, nil, nil, nil)

	bad := inference.OutcomeTimedOut
	failed := &scriptedReviewer{
		script:  make([][]inference.ReviewFinding, 5),
		outcome: []inference.Outcome{inference.OutcomeOK, bad, bad, bad, inference.OutcomeOK},
	}
	incomplete := RunSampled(context.Background(), failed, "{files}", oneCase(seeded), countStub, 3, 1, nil)
	if _, err := Mix(oneCase(seeded), good, incomplete); err == nil || !strings.Contains(err.Error(), "did not complete") {
		t.Errorf("incomplete secondary: err = %v", err)
	}

	other := []Case{{Name: "elsewhere", Files: fixtureFiles()}}
	if _, err := Mix(other, good, good); err == nil || !strings.Contains(err.Error(), "elsewhere") {
		t.Errorf("case in neither run: err = %v", err)
	}

	thin := good
	thin.Cases = []SampledCase{{Case: "c", Samples: good.Cases[0].Samples[:1], Views: good.Cases[0].Views}}
	if _, err := Mix(oneCase(seeded), good, thin); err == nil || !strings.Contains(err.Error(), "at least 2") {
		t.Errorf("one-sample secondary: err = %v", err)
	}
}
