package onboarding

import (
	"context"
	"strings"
	"testing"
)

func TestStatusOkWithStrayAndForeign(t *testing.T) {
	admin := strayAdmin()
	admin.hooks["PROJ"] = []map[string]any{goodHook(nil)}
	row, err := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading("PROJ"), Options{}).
		Status(context.Background(), Target{Project: "PROJ"})
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !row.Owned || !row.BotCanRead || row.Webhook != "ok" {
		t.Errorf("row = %+v", row)
	}
	if strings.Join(row.Stray, ",") != "PROJ/my-repo" {
		t.Errorf("stray = %v", row.Stray)
	}
	if strings.Join(row.Foreign, ",") != "PROJ/other -> https://old.test/webhook/p" {
		t.Errorf("foreign = %v", row.Foreign)
	}
	// Foreign alone does not spoil health; the stray hook does.
	if StatusHealthy([]StatusRow{row}) {
		t.Error("a row with a stray hook reported healthy")
	}
}

// The hook verdict stands when only the repo listing fails; the row is still
// not healthy, and says why.
func TestStatusRepoListingFailureAnnotatesTheVerdict(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"http", statusErr(500, "boom"), "ok (repo hooks unchecked: HTTP 500)"},
		{"transport", transportErr("dial tcp: timeout"), "ok (repo hooks unchecked: dial tcp: timeout)"},
	}
	for _, c := range cases {
		admin := strayAdmin()
		admin.hooks["PROJ"] = []map[string]any{goodHook(nil)}
		admin.reposErr = map[string]error{"PROJ": c.err}
		row, err := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading("PROJ"), Options{}).
			Status(context.Background(), Target{Project: "PROJ"})
		if err != nil {
			t.Fatalf("%s: Status: %v", c.name, err)
		}
		if row.Webhook != c.want {
			t.Errorf("%s: webhook = %q, want %q", c.name, row.Webhook, c.want)
		}
		if len(row.Stray) != 0 {
			t.Errorf("%s: stray = %v, want none", c.name, row.Stray)
		}
	}
}

func TestStatusVerdicts(t *testing.T) {
	stale := goodHook(map[string]any{"events": []any{"pr:opened"}, "configuration": map[string]any{}})
	cases := []struct {
		name string
		hook []map[string]any
		err  error
		want string
		sub  bool
	}{
		{name: "missing", hook: []map[string]any{}, want: "missing"},
		{name: "foreign", hook: []map[string]any{goodHook(map[string]any{"url": "https://old.test/webhook/x"})},
			want: "foreign: https://old.test/webhook/x"},
		{name: "stale", hook: []map[string]any{stale},
			want: "stale: events: missing=", sub: true},
		{name: "http 401", err: statusErr(401, "not permitted"), want: "HTTP 401"},
		{name: "transport", err: transportErr("dial tcp: timeout"), want: "error: dial tcp: timeout"},
	}
	for _, c := range cases {
		admin := &fakeAdmin{
			hooks:   map[string][]map[string]any{"PROJ": {}},
			hookErr: map[string]error{},
		}
		if c.err != nil {
			admin.hookErr["PROJ/my-repo"] = c.err
		} else {
			admin.hooks["PROJ/my-repo"] = c.hook
		}
		row, err := newOnboarder(t, newTeam(repos("PROJ", "my-repo")), admin, botReading("PROJ/my-repo"), Options{}).
			Status(context.Background(), Target{Project: "PROJ", Repo: "my-repo"})
		if err != nil {
			t.Fatalf("%s: Status: %v", c.name, err)
		}
		ok := row.Webhook == c.want
		if c.sub {
			ok = strings.HasPrefix(row.Webhook, c.want) &&
				strings.Contains(row.Webhook, "configuration.secret: (unset) -> (set)")
		}
		if !ok {
			t.Errorf("%s: webhook = %q, want %q", c.name, row.Webhook, c.want)
		}
	}
}

func TestStatusNotOwnedAndWholeClaimBlock(t *testing.T) {
	admin := &fakeAdmin{hooks: map[string][]map[string]any{"PROJ": {}}}
	o := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading("PROJ", "PROJ/my-repo"), Options{})

	other, err := o.Status(context.Background(), Target{Project: "OTHER"})
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if other.Owned || other.BotCanRead || !strings.Contains(other.Webhook, "ask the noergler admin") {
		t.Errorf("unowned row = %+v", other)
	}

	// A repo target under a whole-project claim would double-deliver.
	blocked, err := o.Status(context.Background(), Target{Project: "PROJ", Repo: "my-repo"})
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if !blocked.Owned || !strings.HasPrefix(blocked.Webhook, "blocked: teams.yaml claims all of PROJ") {
		t.Errorf("blocked row = %+v", blocked)
	}
}

// A Bitbucket hiccup on the guard is an error, not a licence to
// double-hook.
func TestGuardFailureIsReportedNotIgnored(t *testing.T) {
	ctx := context.Background()
	tm := newTeam(repos("PROJ", "my-repo"))
	target := Target{Project: "PROJ", Repo: "my-repo"}

	// 403 on the guard: no project admin, the repo verdict stands.
	admin := &fakeAdmin{
		hooks:   map[string][]map[string]any{"PROJ/my-repo": {}},
		hookErr: map[string]error{"PROJ": statusErr(403, "no")},
	}
	row, err := newOnboarder(t, tm, admin, botReading("PROJ/my-repo"), Options{}).Status(ctx, target)
	if err != nil || row.Webhook != "missing" {
		t.Errorf("403 guard: webhook = %q, err = %v", row.Webhook, err)
	}

	// 503 on the guard: onboard fails the target, naming the status.
	admin = &fakeAdmin{
		hooks:   map[string][]map[string]any{"PROJ/my-repo": {}},
		hookErr: map[string]error{"PROJ": statusErr(503, "unavailable")},
	}
	o := newOnboarder(t, tm, admin, botReading("PROJ/my-repo"), Options{})
	res, err := o.Onboard(ctx, target)
	if err != nil {
		t.Fatalf("Onboard: %v", err)
	}
	if res.Status != "failed" || !strings.Contains(res.Detail, "HTTP 503") {
		t.Errorf("result = %+v", res)
	}

	// Status maps the same failure to a row of its own.
	row, err = o.Status(ctx, target)
	if err != nil || row.Webhook != "project hook check HTTP 503" {
		t.Errorf("503 status row: webhook = %q, err = %v", row.Webhook, err)
	}
}

// A transport failure in the guard is not a verdict: it escapes to Run,
// which turns it into an error row, and the status row loses its claim:
// owned and bot both come back false.
func TestTransportFailureInTheGuardEscapesToRun(t *testing.T) {
	ctx := context.Background()
	tm := newTeam(repos("PROJ", "my-repo"))
	target := Target{Project: "PROJ", Repo: "my-repo"}
	admin := &fakeAdmin{
		hooks:   map[string][]map[string]any{"PROJ/my-repo": {}},
		hookErr: map[string]error{"PROJ": transportErr("dial tcp: no route to host")},
	}
	o := newOnboarder(t, tm, admin, botReading("PROJ/my-repo"), Options{})

	if _, err := o.Status(ctx, target); err == nil {
		t.Fatal("Status swallowed the transport failure")
	}
	rows, _, _, healthy := Run(ctx, o, ActionStatus, []Target{target})
	if len(rows) != 1 {
		t.Fatalf("rows = %v", rows)
	}
	if rows[0].Owned || rows[0].BotCanRead {
		t.Errorf("row kept its claim through the blanket except: %+v", rows[0])
	}
	if rows[0].Webhook != "error: dial tcp: no route to host" {
		t.Errorf("webhook = %q", rows[0].Webhook)
	}
	if healthy {
		t.Error("an error row reported healthy")
	}

	if _, err := o.Onboard(ctx, target); err == nil {
		t.Fatal("Onboard swallowed the transport failure")
	}
	_, results, _, healthy := Run(ctx, o, ActionOnboard, []Target{target})
	if results[0].Status != "failed" ||
		results[0].Detail != "unexpected error: dial tcp: no route to host" {
		t.Errorf("result = %+v", results[0])
	}
	if healthy {
		t.Error("a failed row reported healthy")
	}
}

// A transport failure in the grant escapes to Run the same way.
func TestTransportFailureInTheGrantEscapesToRun(t *testing.T) {
	admin := &fakeAdmin{
		hooks:   map[string][]map[string]any{"PROJ": {}},
		hookErr: map[string]error{"grant:PROJ": transportErr("dial tcp: timeout")},
		repos:   map[string][]map[string]any{"PROJ": nil},
	}
	o := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading(), Options{GrantBot: true})
	if _, err := o.Onboard(context.Background(), Target{Project: "PROJ"}); err == nil {
		t.Fatal("Onboard swallowed the transport failure in the grant")
	}
	_, results, _, _ := Run(context.Background(), o, ActionGrantBot, []Target{{Project: "PROJ"}})
	if !strings.HasPrefix(results[0].Detail, "unexpected error: ") {
		t.Errorf("result = %+v", results[0])
	}

	// An HTTP failure in the grant stays inside Onboard, with the body.
	admin.hookErr["grant:PROJ"] = statusErr(403, "not permitted")
	res, err := o.Onboard(context.Background(), Target{Project: "PROJ"})
	if err != nil {
		t.Fatalf("Onboard: %v", err)
	}
	want := "grant PROJECT_WRITE to noergler: HTTP 403: not permitted"
	if res.Status != "failed" || res.Detail != want {
		t.Errorf("detail = %q, want %q", res.Detail, want)
	}
}

func TestOnboardCreateWithGrantAndPrune(t *testing.T) {
	admin := strayAdmin()
	admin.hooks["PROJ"] = []map[string]any{}
	// The bot cannot read the project yet, so the grant runs first.
	res, err := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading(), Options{GrantBot: true}).
		Onboard(context.Background(), Target{Project: "PROJ"})
	if err != nil {
		t.Fatalf("Onboard: %v", err)
	}
	want := "webhook created (id=42), noergler granted PROJECT_WRITE, pruned 1 repo hook(s): PROJ/my-repo"
	if res.Status != "ok" || res.Detail != want {
		t.Errorf("detail =\n%q\nwant\n%q", res.Detail, want)
	}
	if strings.Join(admin.grants, ",") != "PROJ:noergler:PROJECT_WRITE" {
		t.Errorf("grants = %v", admin.grants)
	}
	if strings.Join(admin.deleted, ",") != "PROJ/my-repo#8" {
		t.Errorf("deleted = %v", admin.deleted)
	}
}

func TestOnboardWithoutGrantBotIsSkipped(t *testing.T) {
	admin := &fakeAdmin{hooks: map[string][]map[string]any{"PROJ": {}}}
	res, err := newOnboarder(t, newTeam(whole("PROJ")), admin, &fakeBot{err: statusErr(403, "no")}, Options{}).
		Onboard(context.Background(), Target{Project: "PROJ"})
	if err != nil {
		t.Fatalf("Onboard: %v", err)
	}
	want := "noergler cannot read it; grant noergler PROJECT_WRITE in Bitbucket or run grant-bot"
	if res.Status != "skipped" || res.Detail != want {
		t.Errorf("detail = %q, want %q", res.Detail, want)
	}
	if len(admin.created) != 0 {
		t.Error("a skipped target was hooked anyway")
	}
}

func TestOnboardUpToDateUpdateAndForeign(t *testing.T) {
	ctx := context.Background()
	tm := newTeam(repos("PROJ", "my-repo"))
	target := Target{Project: "PROJ", Repo: "my-repo"}
	cases := []struct {
		name   string
		hook   map[string]any
		status string
		detail string
		diff   string
	}{
		{"up to date", goodHook(nil), "ok", "webhook already up to date", ""},
		{"stale", goodHook(map[string]any{"active": false}), "ok", "webhook updated (id=7)", "active: False -> True"},
		{"foreign", goodHook(map[string]any{"url": "https://old.test/webhook"}), "failed",
			"'noergler' hook points at another noergler (https://old.test/webhook); gone? remove it with that instance first, else use another name", ""},
	}
	for _, c := range cases {
		admin := &fakeAdmin{hooks: map[string][]map[string]any{
			"PROJ": {}, "PROJ/my-repo": {c.hook}}}
		res, err := newOnboarder(t, tm, admin, botReading("PROJ/my-repo"), Options{NoPrune: true}).Onboard(ctx, target)
		if err != nil {
			t.Fatalf("%s: Onboard: %v", c.name, err)
		}
		if res.Status != c.status || res.Detail != c.detail {
			t.Errorf("%s: %s %q, want %s %q", c.name, res.Status, res.Detail, c.status, c.detail)
		}
		if strings.Join(res.Diff, ",") != c.diff {
			t.Errorf("%s: diff = %v, want %q", c.name, res.Diff, c.diff)
		}
	}
}

func TestOnboardDryRunWritesNothing(t *testing.T) {
	admin := &fakeAdmin{
		hooks: map[string][]map[string]any{"PROJ": {}},
		repos: map[string][]map[string]any{"PROJ": nil},
	}
	res, err := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading(), Options{GrantBot: true, DryRun: true}).
		Onboard(context.Background(), Target{Project: "PROJ"})
	if err != nil {
		t.Fatalf("Onboard: %v", err)
	}
	if res.Status != "ok" || !strings.HasPrefix(res.Detail, "dry-run: webhook created (id=-1)") {
		t.Errorf("result = %+v", res)
	}
	if len(admin.created) != 0 || len(admin.updated) != 0 || len(admin.deleted) != 0 || len(admin.grants) != 0 {
		t.Errorf("dry run wrote: created=%v updated=%v deleted=%v grants=%v",
			admin.created, admin.updated, admin.deleted, admin.grants)
	}
}

// A failed prune annotates the result but never fails it.
func TestOnboardPruneFailureIsANoteNotAFailure(t *testing.T) {
	admin := &fakeAdmin{
		hooks:    map[string][]map[string]any{"PROJ": {goodHook(nil)}},
		reposErr: map[string]error{"PROJ": statusErr(500, "boom")},
	}
	res, err := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading("PROJ"), Options{}).
		Onboard(context.Background(), Target{Project: "PROJ"})
	if err != nil {
		t.Fatalf("Onboard: %v", err)
	}
	if res.Status != "ok" || !strings.Contains(res.Detail, "prune of repo hooks failed:") {
		t.Errorf("result = %+v", res)
	}
}

func TestRemove(t *testing.T) {
	ctx := context.Background()
	target := Target{Project: "PROJ"}
	cases := []struct {
		name   string
		hooks  []map[string]any
		err    error
		status string
		detail string
		delete string
	}{
		{name: "ours", hooks: []map[string]any{goodHook(nil)}, status: "ok",
			detail: "webhook removed: id=7", delete: "PROJ#7"},
		{name: "foreign", hooks: []map[string]any{goodHook(map[string]any{"url": "https://old.test/webhook"})},
			status: "skipped",
			detail: "'noergler' webhook points at another noergler (https://old.test/webhook), left alone"},
		{name: "absent", hooks: []map[string]any{}, status: "skipped",
			detail: "no 'noergler' webhook found"},
		{name: "list fails", err: statusErr(500, "boom"), status: "failed",
			detail: "list webhooks HTTP 500: boom"},
		{name: "list unreachable", err: transportErr("dial tcp: timeout"), status: "failed",
			detail: "list webhooks: dial tcp: timeout"},
	}
	for _, c := range cases {
		admin := &fakeAdmin{hooks: map[string][]map[string]any{}, hookErr: map[string]error{}}
		if c.err != nil {
			admin.hookErr["PROJ"] = c.err
		} else {
			admin.hooks["PROJ"] = c.hooks
		}
		res, err := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading("PROJ"), Options{}).Remove(ctx, target)
		if err != nil {
			t.Fatalf("%s: Remove: %v", c.name, err)
		}
		if res.Status != c.status || res.Detail != c.detail {
			t.Errorf("%s: %s %q, want %s %q", c.name, res.Status, res.Detail, c.status, c.detail)
		}
		if strings.Join(admin.deleted, ",") != c.delete {
			t.Errorf("%s: deleted = %v, want %q", c.name, admin.deleted, c.delete)
		}
	}
}

// A dry-run remove reports the id it would have deleted and deletes nothing.
func TestRemoveDryRun(t *testing.T) {
	admin := &fakeAdmin{hooks: map[string][]map[string]any{"PROJ": {goodHook(nil)}}}
	res, err := newOnboarder(t, newTeam(whole("PROJ")), admin, botReading("PROJ"), Options{DryRun: true}).
		Remove(context.Background(), Target{Project: "PROJ"})
	if err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if res.Status != "ok" || res.Detail != "dry-run: would remove webhook id=7" {
		t.Errorf("result = %+v", res)
	}
	if len(admin.deleted) != 0 {
		t.Errorf("dry run deleted %v", admin.deleted)
	}
}

func TestRunOneFailureDoesNotAbortTheRest(t *testing.T) {
	tm := newTeam(repos("PROJ", "my-repo", "other"))
	admin := &fakeAdmin{
		hooks: map[string][]map[string]any{
			"PROJ":       {},
			"PROJ/other": {goodHook(nil)},
		},
		hookErr: map[string]error{"PROJ/my-repo": statusErr(401, "no")},
	}
	targets, err := TargetsFor(tm, nil)
	if err != nil {
		t.Fatalf("TargetsFor: %v", err)
	}
	o := newOnboarder(t, tm, admin, botReading("PROJ/my-repo", "PROJ/other"), Options{})
	_, results, text, healthy := Run(context.Background(), o, ActionOnboard, targets)

	if len(results) != 2 || results[0].Status != "failed" || results[1].Status != "ok" {
		t.Fatalf("results = %+v", results)
	}
	if healthy {
		t.Error("a failed row reported healthy")
	}
	if text != RenderResults(results) {
		t.Error("text does not match the rendered results")
	}
	if !strings.Contains(text, "PROJ/my-repo") || !strings.Contains(text, "HTTP 401") {
		t.Errorf("text =\n%s", text)
	}
}

// Run keeps the target order it was given, and is sequential.
func TestRunIsSequentialAndOrdered(t *testing.T) {
	tm := newTeam(repos("PROJ", "a", "b", "c"))
	admin := &fakeAdmin{hooks: map[string][]map[string]any{
		"PROJ": {}, "PROJ/a": {}, "PROJ/b": {}, "PROJ/c": {}}}
	targets, _ := TargetsFor(tm, nil)
	o := newOnboarder(t, tm, admin, botReading("PROJ/a", "PROJ/b", "PROJ/c"), Options{})
	_, results, _, _ := Run(context.Background(), o, ActionOnboard, targets)
	for i, want := range []string{"a", "b", "c"} {
		if results[i].Target.Repo != want {
			t.Errorf("result %d = %s, want %s", i, results[i].Target.Repo, want)
		}
	}
	if admin.maxInFlight.Load() != 1 {
		t.Errorf("Run overlapped %d listings; it must be sequential", admin.maxInFlight.Load())
	}
}

// Run over no targets is empty, rendered and vacuously healthy.
func TestRunWithNoTargets(t *testing.T) {
	o := newOnboarder(t, newTeam(), &fakeAdmin{}, botReading(), Options{})
	rows, _, text, healthy := Run(context.Background(), o, ActionStatus, nil)
	if len(rows) != 0 || !healthy || text != RenderStatus(nil) {
		t.Errorf("status: rows=%v healthy=%t", rows, healthy)
	}
	_, results, text, healthy := Run(context.Background(), o, ActionRemove, nil)
	if len(results) != 0 || !healthy || text != RenderResults(nil) {
		t.Errorf("remove: results=%v healthy=%t", results, healthy)
	}
}

// httpDetail carries the response body, capped at 200 runes.
func TestHTTPDetail(t *testing.T) {
	if got := httpDetail(statusErr(401, "not permitted")); got != "HTTP 401: not permitted" {
		t.Errorf("httpDetail = %q", got)
	}
	long := httpDetail(statusErr(500, strings.Repeat("x", 500)))
	if len(long) != len("HTTP 500: ")+200 {
		t.Errorf("body not capped at 200: %d chars", len(long))
	}
}
