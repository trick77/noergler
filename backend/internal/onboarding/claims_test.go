package onboarding

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/store"
)

// The real store must keep satisfying the consumer-side claim interface.
var _ ClaimStore = (*store.Store)(nil)

// fakeStore records the claim calls and answers from canned values.
type fakeStore struct {
	claims    []config.ProjectScope
	addErr    error
	added     []config.ProjectScope
	removed   []config.ProjectScope
	purged    []string
	purgeN    map[string]int
	countCall []string
}

func (f *fakeStore) AddClaims(_ context.Context, _ string, scopes []config.ProjectScope, _ string) ([]string, error) {
	if f.addErr != nil {
		return nil, f.addErr
	}
	f.added = append(f.added, scopes...)
	f.claims = append(f.claims, scopes...)
	out := make([]string, 0, len(scopes))
	for _, s := range scopes {
		out = append(out, s.String())
	}
	return out, nil
}

func (f *fakeStore) RemoveClaims(_ context.Context, _ string, scopes []config.ProjectScope) ([]string, error) {
	f.removed = append(f.removed, scopes...)
	f.claims = nil
	out := make([]string, 0, len(scopes))
	for _, s := range scopes {
		out = append(out, s.String())
	}
	return out, nil
}

func (f *fakeStore) ListClaims(context.Context, string) ([]config.ProjectScope, error) {
	return f.claims, nil
}

func key(project string, repo *string) string {
	if repo == nil {
		return project
	}
	return project + "/" + *repo
}

func (f *fakeStore) PurgeProject(_ context.Context, _, project string, repo *string) (int, error) {
	k := key(project, repo)
	f.purged = append(f.purged, k)
	return f.purgeN[k], nil
}

func (f *fakeStore) CountProjectPRs(_ context.Context, _, project string, repo *string) (int, error) {
	k := key(project, repo)
	f.countCall = append(f.countCall, k)
	return f.purgeN[k], nil
}

// HasAdmin is not a boolean: only 401/403 are a plain "no". Everything else,
// 404 and every transport failure included, is Bitbucket's fault and aborts
// the whole request.
func TestHasAdmin(t *testing.T) {
	ctx := context.Background()
	target := Target{Project: "PROJ", Repo: "r"}
	cases := []struct {
		name     string
		err      error
		ok       bool
		upstream bool
		msg      string
	}{
		{name: "listing works", ok: true},
		{name: "401", err: statusErr(401, "x")},
		{name: "403", err: statusErr(403, "x")},
		{name: "404 is upstream, not a no", err: statusErr(404, "x"), upstream: true,
			msg: "Bitbucket answered HTTP 404 on PROJ/r"},
		{name: "500", err: statusErr(500, "boom"), upstream: true,
			msg: "Bitbucket answered HTTP 500 on PROJ/r"},
		{name: "transport", err: transportErr("dial tcp: timeout"), upstream: true,
			msg: "Bitbucket unreachable on PROJ/r: dial tcp: timeout"},
	}
	for _, c := range cases {
		admin := &fakeAdmin{hooks: map[string][]map[string]any{"PROJ/r": {}}, hookErr: map[string]error{}}
		if c.err != nil {
			admin.hookErr["PROJ/r"] = c.err
		}
		ok, err := HasAdmin(ctx, admin, target)
		var ue *UpstreamError
		switch {
		case c.upstream:
			if !errors.As(err, &ue) {
				t.Errorf("%s: err = %v, want *UpstreamError", c.name, err)
				continue
			}
			if ue.Error() != c.msg {
				t.Errorf("%s: message = %q, want %q", c.name, ue.Error(), c.msg)
			}
		default:
			if err != nil {
				t.Errorf("%s: unexpected err %v", c.name, err)
			}
			if ok != c.ok {
				t.Errorf("%s: ok = %t, want %t", c.name, ok, c.ok)
			}
		}
	}
}

func claimAdmin() *fakeAdmin {
	return &fakeAdmin{
		hooks:   map[string][]map[string]any{"PROJ": {}, "PROJ/a": {}, "PROJ/b": {}},
		hookErr: map[string]error{},
		repos:   map[string][]map[string]any{"PROJ": nil},
	}
}

// The proven scopes are claimed, then hooked; the team's projects become the
// fresh claim list.
func TestClaimAndOnboardClaimsWhatIsProven(t *testing.T) {
	admin := claimAdmin()
	st := &fakeStore{}
	tm := newTeam()
	res, err := ClaimAndOnboard(context.Background(), admin, botReading("PROJ"), st, tm,
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	if err != nil {
		t.Fatalf("ClaimAndOnboard: %v", err)
	}
	if strings.Join(res.Claimed, ",") != "PROJ" {
		t.Errorf("claimed = %v", res.Claimed)
	}
	if len(res.Results) != 1 || res.Results[0].Status != "ok" || !res.Healthy {
		t.Errorf("results = %+v healthy=%t", res.Results, res.Healthy)
	}
	if len(res.Claims) != 1 || res.Claims[0].Key != "PROJ" {
		t.Errorf("claims = %v, want the fresh claim list", res.Claims)
	}
	// The caller's team is never mutated; it applies res.Claims itself.
	if len(tm.Projects) != 0 {
		t.Errorf("team was mutated: %v", tm.Projects)
	}
	if len(admin.created) != 1 {
		t.Errorf("created %d webhooks, want 1", len(admin.created))
	}
	if res.Text != RenderResults(res.Results) {
		t.Error("text does not match the results")
	}
}

// A target without admin is not claimed and not hooked, and its failed row is
// PREPENDED to the results.
func TestClaimAndOnboardFailedRowsComeFirst(t *testing.T) {
	admin := claimAdmin()
	admin.hookErr["PROJ/a"] = statusErr(403, "no")
	st := &fakeStore{}
	tm := newTeam()
	res, err := ClaimAndOnboard(context.Background(), admin, botReading("PROJ/a", "PROJ/b"), st, tm,
		[]config.ProjectScope{repos("PROJ", "a", "b")}, "team:platform", webhookURL, Options{}, nil)
	if err != nil {
		t.Fatalf("ClaimAndOnboard: %v", err)
	}
	if len(res.Results) != 2 {
		t.Fatalf("results = %+v", res.Results)
	}
	first := res.Results[0]
	if first.Status != "failed" || first.Target.Repo != "a" ||
		first.Detail != "no project admin on PROJ/a with this token; not claimed" {
		t.Errorf("first result = %+v, want the unprovable target", first)
	}
	if res.Results[1].Status != "ok" || res.Results[1].Target.Repo != "b" {
		t.Errorf("second result = %+v", res.Results[1])
	}
	// Only the proven repo was claimed.
	if len(st.added) != 1 || strings.Join(st.added[0].Repos, ",") != "b" {
		t.Errorf("added = %v, want only PROJ/b", st.added)
	}
	if res.Healthy {
		t.Error("a failed row reported healthy")
	}
}

// Nothing is proven: nothing is claimed, nothing is hooked, and the failed
// rows are all there is.
func TestClaimAndOnboardNothingProven(t *testing.T) {
	admin := claimAdmin()
	admin.hookErr["PROJ"] = statusErr(401, "no")
	st := &fakeStore{}
	res, err := ClaimAndOnboard(context.Background(), admin, botReading("PROJ"), st, newTeam(),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	if err != nil {
		t.Fatalf("ClaimAndOnboard: %v", err)
	}
	if len(res.Results) != 1 || res.Results[0].Status != "failed" {
		t.Errorf("results = %+v", res.Results)
	}
	if len(st.added) != 0 || len(admin.created) != 0 {
		t.Error("an unprovable target was claimed or hooked")
	}
	if len(res.Claimed) != 0 {
		t.Errorf("claimed = %v, want none", res.Claimed)
	}
}

// An UpstreamError from the admin proof aborts the whole request: no claim, no
// hook, no partial result.
func TestClaimAndOnboardUpstreamErrorAborts(t *testing.T) {
	admin := claimAdmin()
	admin.hookErr["PROJ/b"] = statusErr(500, "boom")
	st := &fakeStore{}
	_, err := ClaimAndOnboard(context.Background(), admin, botReading("PROJ/a", "PROJ/b"), st, newTeam(),
		[]config.ProjectScope{repos("PROJ", "a", "b")}, "team:platform", webhookURL, Options{}, nil)
	var ue *UpstreamError
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *UpstreamError", err)
	}
	if len(st.added) != 0 || len(admin.created) != 0 {
		t.Error("the request wrote something before aborting")
	}
}

// A dry run claims nothing but still hooks against the proven scopes, which
// count as claimed for the ownership check.
func TestClaimAndOnboardDryRunClaimsNothingButReports(t *testing.T) {
	admin := claimAdmin()
	st := &fakeStore{}
	tm := newTeam()
	res, err := ClaimAndOnboard(context.Background(), admin, botReading("PROJ"), st, tm,
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{DryRun: true}, nil)
	if err != nil {
		t.Fatalf("ClaimAndOnboard: %v", err)
	}
	if len(st.added) != 0 {
		t.Errorf("dry run claimed %v", st.added)
	}
	if len(res.Claimed) != 0 {
		t.Errorf("claimed = %v, want none", res.Claimed)
	}
	// The hook step still ran, against the unclaimed-but-proven scope.
	if len(res.Results) != 1 || res.Results[0].Status != "ok" ||
		!strings.HasPrefix(res.Results[0].Detail, "dry-run: ") {
		t.Errorf("results = %+v", res.Results)
	}
	if len(admin.created) != 0 {
		t.Error("dry run wrote to Bitbucket")
	}
	// A request that wrote nothing must not leave the caller's team holding
	// phantom claims.
	if len(tm.Projects) != 0 {
		t.Errorf("dry run mutated the team: %v", tm.Projects)
	}
	if res.Claims != nil {
		t.Errorf("dry run reported claims %v", res.Claims)
	}
}

// A ClaimConflict comes back unwrapped, for the HTTP layer's 409.
func TestClaimAndOnboardSurfacesAClaimConflict(t *testing.T) {
	repo := "a"
	conflict := &store.ClaimConflict{Project: "PROJ", Repo: &repo, OtherTeam: "other"}
	admin := claimAdmin()
	st := &fakeStore{addErr: conflict}
	_, err := ClaimAndOnboard(context.Background(), admin, botReading("PROJ"), st, newTeam(),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	var cc *store.ClaimConflict
	if !errors.As(err, &cc) || cc.OtherTeam != "other" {
		t.Fatalf("err = %v, want the ClaimConflict", err)
	}
	if len(admin.created) != 0 {
		t.Error("a conflicting claim was hooked anyway")
	}
}

// grant-bot is the action the hook step runs under when GrantBot is set.
func TestClaimAndOnboardGrantBotAction(t *testing.T) {
	admin := claimAdmin()
	res, err := ClaimAndOnboard(context.Background(), admin, botReading(), &fakeStore{}, newTeam(),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{GrantBot: true}, nil)
	if err != nil {
		t.Fatalf("ClaimAndOnboard: %v", err)
	}
	if strings.Join(admin.grants, ",") != "PROJ:noergler:PROJECT_WRITE" {
		t.Errorf("grants = %v", admin.grants)
	}
	if res.Results[0].Status != "ok" {
		t.Errorf("result = %+v", res.Results[0])
	}
}

func removeAdmin() *fakeAdmin {
	return &fakeAdmin{
		hooks: map[string][]map[string]any{
			"PROJ":   {goodHook(nil)},
			"PROJ/a": {goodHook(map[string]any{"id": float64(11)})},
			"PROJ/b": {goodHook(map[string]any{"id": float64(12)})},
		},
		hookErr: map[string]error{},
	}
}

// A whole-project scope gives up every claim the team holds on that project.
func TestRemoveAndUnclaimWholeScopeCoversRepoClaims(t *testing.T) {
	admin := removeAdmin()
	st := &fakeStore{purgeN: map[string]int{"PROJ/a": 3, "PROJ/b": 4}}
	tm := newTeam(repos("PROJ", "a", "b"))
	res, err := RemoveAndUnclaim(context.Background(), admin, botReading(), st, tm,
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	if err != nil {
		t.Fatalf("RemoveAndUnclaim: %v", err)
	}
	if len(res.Results) != 2 {
		t.Fatalf("results = %+v", res.Results)
	}
	for i, want := range []string{
		"webhook removed: id=11; purged 3 PR record(s)",
		"webhook removed: id=12; purged 4 PR record(s)",
	} {
		if res.Results[i].Detail != want {
			t.Errorf("result %d detail = %q, want %q", i, res.Results[i].Detail, want)
		}
	}
	if res.PurgedPRs != 7 {
		t.Errorf("purged = %d, want 7", res.PurgedPRs)
	}
	if strings.Join(admin.deleted, ",") != "PROJ/a#11,PROJ/b#12" {
		t.Errorf("deleted = %v", admin.deleted)
	}
	if len(st.removed) != 2 {
		t.Errorf("removed claims = %v", st.removed)
	}
	if len(res.Claims) != 0 {
		t.Errorf("claims = %v, want none left", res.Claims)
	}
	// The caller's team is never mutated; it applies res.Claims itself.
	if len(tm.Projects) != 1 {
		t.Errorf("team was mutated: %v", tm.Projects)
	}
}

// PR purging runs for EVERY hook result, the failed ones included.
func TestRemoveAndUnclaimPurgesEvenAFailedHookRemoval(t *testing.T) {
	admin := removeAdmin()
	// Admin proves out on both, but deleting PROJ/b's hook fails.
	admin.deleteFn = func(_, repo string, _ int) error {
		if repo == "b" {
			return statusErr(500, "boom")
		}
		return nil
	}
	st := &fakeStore{purgeN: map[string]int{"PROJ/a": 1, "PROJ/b": 5}}
	res, err := RemoveAndUnclaim(context.Background(), admin, botReading(), st,
		newTeam(repos("PROJ", "a", "b")),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	if err != nil {
		t.Fatalf("RemoveAndUnclaim: %v", err)
	}
	// PROJ/b's hook removal failed, but its records still went.
	var failed *TargetResult
	for i := range res.Results {
		if res.Results[i].Status == "failed" {
			failed = &res.Results[i]
		}
	}
	if failed == nil {
		t.Fatalf("no failed row in %+v", res.Results)
	}
	if !strings.HasSuffix(failed.Detail, "; purged 5 PR record(s)") {
		t.Errorf("failed row = %q, want the purge appended", failed.Detail)
	}
	if strings.Join(st.purged, ",") != "PROJ/a,PROJ/b" {
		t.Errorf("purged = %v, want both", st.purged)
	}
	if res.PurgedPRs != 6 {
		t.Errorf("purged = %d, want 6", res.PurgedPRs)
	}
	if res.Healthy {
		t.Error("a failed row reported healthy")
	}
}

// Without admin on a target nothing is removed, nothing purged for it, and
// nothing unclaimed.
func TestRemoveAndUnclaimNeedsAdmin(t *testing.T) {
	admin := removeAdmin()
	admin.hookErr["PROJ/a"] = statusErr(403, "no")
	st := &fakeStore{purgeN: map[string]int{"PROJ/b": 2}}
	res, err := RemoveAndUnclaim(context.Background(), admin, botReading(), st,
		newTeam(repos("PROJ", "a", "b")),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	if err != nil {
		t.Fatalf("RemoveAndUnclaim: %v", err)
	}
	if res.Results[0].Status != "failed" ||
		res.Results[0].Detail != "no project admin on PROJ/a with this token; not removed" {
		t.Errorf("first result = %+v", res.Results[0])
	}
	// The unprovable target never reached the purge loop.
	if strings.Join(st.purged, ",") != "PROJ/b" {
		t.Errorf("purged = %v, want only PROJ/b", st.purged)
	}
	if len(st.removed) != 1 || strings.Join(st.removed[0].Repos, ",") != "b" {
		t.Errorf("unclaimed = %v, want only PROJ/b", st.removed)
	}
}

// A dry run counts the records instead of purging them and gives nothing up.
func TestRemoveAndUnclaimDryRunCountsOnly(t *testing.T) {
	admin := removeAdmin()
	st := &fakeStore{purgeN: map[string]int{"PROJ": 9}}
	tm := newTeam(whole("PROJ"))
	res, err := RemoveAndUnclaim(context.Background(), admin, botReading(), st, tm,
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{DryRun: true}, nil)
	if err != nil {
		t.Fatalf("RemoveAndUnclaim: %v", err)
	}
	want := "dry-run: would remove webhook id=7; dry-run: would purge 9 PR record(s)"
	if res.Results[0].Detail != want {
		t.Errorf("detail = %q, want %q", res.Results[0].Detail, want)
	}
	if res.PurgedPRs != 9 {
		t.Errorf("purged = %d, want 9", res.PurgedPRs)
	}
	if len(st.purged) != 0 || len(st.removed) != 0 || len(admin.deleted) != 0 {
		t.Errorf("dry run wrote: purged=%v removed=%v deleted=%v", st.purged, st.removed, admin.deleted)
	}
	if len(tm.Projects) != 1 {
		t.Errorf("dry run changed the team's claims: %v", tm.Projects)
	}
}

// A whole-project scope on a project the team holds nothing on is a NoClaim.
func TestRemoveAndUnclaimRejectsAnUnclaimedProject(t *testing.T) {
	_, err := RemoveAndUnclaim(context.Background(), removeAdmin(), botReading(), &fakeStore{},
		newTeam(whole("PROJ")),
		[]config.ProjectScope{whole("OTHER")}, "team:platform", webhookURL, Options{}, nil)
	var nc *NoClaim
	if !errors.As(err, &nc) || nc.Error() != "no claim on OTHER" {
		t.Fatalf("err = %v, want NoClaim", err)
	}
}

// A repo scope naming a repo the team does not hold is an UnknownTarget.
func TestRemoveAndUnclaimRejectsAnUnknownRepo(t *testing.T) {
	_, err := RemoveAndUnclaim(context.Background(), removeAdmin(), botReading(), &fakeStore{},
		newTeam(repos("PROJ", "a")),
		[]config.ProjectScope{repos("PROJ", "zz")}, "team:platform", webhookURL, Options{}, nil)
	var ut *UnknownTarget
	if !errors.As(err, &ut) || !strings.Contains(ut.Error(), "PROJ/zz") {
		t.Fatalf("err = %v, want UnknownTarget naming PROJ/zz", err)
	}
}

// An UpstreamError from the admin proof aborts the removal before anything is
// deleted or purged.
func TestRemoveAndUnclaimUpstreamErrorAborts(t *testing.T) {
	admin := removeAdmin()
	admin.hookErr["PROJ/b"] = transportErr("dial tcp: timeout")
	st := &fakeStore{}
	_, err := RemoveAndUnclaim(context.Background(), admin, botReading(), st,
		newTeam(repos("PROJ", "a", "b")),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	var ue *UpstreamError
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *UpstreamError", err)
	}
	if ue.Error() != "Bitbucket unreachable on PROJ/b: dial tcp: timeout" {
		t.Errorf("message = %q", ue.Error())
	}
	if len(admin.deleted) != 0 || len(st.purged) != 0 || len(st.removed) != 0 {
		t.Error("the removal wrote something before aborting")
	}
}

// A project-level claim purges the whole project (repo == nil).
func TestRemoveAndUnclaimProjectPurgeUsesANilRepo(t *testing.T) {
	admin := removeAdmin()
	st := &fakeStore{purgeN: map[string]int{"PROJ": 2}}
	if _, err := RemoveAndUnclaim(context.Background(), admin, botReading(), st, newTeam(whole("PROJ")),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil); err != nil {
		t.Fatalf("RemoveAndUnclaim: %v", err)
	}
	if strings.Join(st.purged, ",") != "PROJ" {
		t.Errorf("purged = %v, want the whole project", st.purged)
	}
	if len(st.removed) != 1 || st.removed[0].Repos != nil {
		t.Errorf("removed = %v, want a whole-project scope", st.removed)
	}
}

func TestRemoveAndUnclaimRendersItsTable(t *testing.T) {
	res, err := RemoveAndUnclaim(context.Background(), removeAdmin(), botReading(),
		&fakeStore{purgeN: map[string]int{"PROJ": 0}}, newTeam(whole("PROJ")),
		[]config.ProjectScope{whole("PROJ")}, "team:platform", webhookURL, Options{}, nil)
	if err != nil {
		t.Fatalf("RemoveAndUnclaim: %v", err)
	}
	if res.Text != RenderResults(res.Results) {
		t.Errorf("text =\n%s\nresults =\n%s", res.Text, RenderResults(res.Results))
	}
	if !strings.Contains(res.Text, fmt.Sprintf("purged %d PR record(s)", 0)) {
		t.Errorf("text =\n%s", res.Text)
	}
}
