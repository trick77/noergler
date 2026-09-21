package onboarding

import (
	"errors"
	"testing"
)

func TestTargetAccessors(t *testing.T) {
	cases := []struct {
		target                    Target
		isProject                 bool
		key, label, botPermission string
	}{
		{Target{Project: "PROJ"}, true, "PROJ", "PROJ (project)", "PROJECT_WRITE"},
		{Target{Project: "PROJ", Repo: "r"}, false, "PROJ/r", "PROJ/r", "REPO_WRITE"},
	}
	for _, c := range cases {
		if got := c.target.IsProject(); got != c.isProject {
			t.Errorf("%v IsProject = %t, want %t", c.target, got, c.isProject)
		}
		if got := c.target.Key(); got != c.key {
			t.Errorf("%v Key = %q, want %q", c.target, got, c.key)
		}
		if got := c.target.Label(); got != c.label {
			t.Errorf("%v Label = %q, want %q", c.target, got, c.label)
		}
		if got := c.target.BotPermission(); got != c.botPermission {
			t.Errorf("%v BotPermission = %q, want %q", c.target, got, c.botPermission)
		}
	}
}

func TestTargetsForWholeAndRepos(t *testing.T) {
	got, err := TargetsFor(newTeam(whole("A"), repos("B", "x", "y")), nil)
	if err != nil {
		t.Fatalf("TargetsFor: %v", err)
	}
	want := []Target{{Project: "A"}, {Project: "B", Repo: "x"}, {Project: "B", Repo: "y"}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("target %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestTargetsForSubsetFiltersAndKeepsOrder(t *testing.T) {
	tm := newTeam(whole("A"), repos("B", "x", "y"))
	got, err := TargetsFor(tm, []string{"B/y", "A"})
	if err != nil {
		t.Fatalf("TargetsFor: %v", err)
	}
	want := []Target{{Project: "B", Repo: "y"}, {Project: "A"}}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("target %d = %v, want %v", i, got[i], want[i])
		}
	}
	// An empty non-nil subset narrows to nothing; nil means all.
	if got, _ := TargetsFor(tm, []string{}); len(got) != 0 {
		t.Errorf("empty subset = %v, want none", got)
	}
}

// The message is pinned byte for byte, duplicates in the unknown list
// included, with `known:` in claim order.
func TestTargetsForUnknownMessage(t *testing.T) {
	_, err := TargetsFor(newTeam(whole("A"), repos("B", "x")), []string{"A", "B/y", "Z", "B/y"})
	var ut *UnknownTarget
	if !errors.As(err, &ut) {
		t.Fatalf("err = %v, want *UnknownTarget", err)
	}
	want := "not in team platform's teams.yaml block: B/y, Z, B/y; known: A, B/x"
	if ut.Error() != want {
		t.Errorf("message =\n%q\nwant\n%q", ut.Error(), want)
	}
}

func TestClaimKind(t *testing.T) {
	tm := newTeam(whole("A"), repos("B", "x"))
	for project, want := range map[string]string{"A": "whole", "B": "repos", "C": "none"} {
		if got := claimKind(tm, project); got != want {
			t.Errorf("claimKind(%s) = %q, want %q", project, got, want)
		}
	}
}

func TestRefusalReasons(t *testing.T) {
	cases := []struct {
		name string
		got  string
		want string
	}{
		{"project under a repos claim", notOwnedReason(Target{Project: "P"}, Claim{Kind: "repos"}),
			"teams.yaml lists specific repos of P for this team; onboard those, or ask the noergler admin to claim the whole project"},
		{"repo under a repos claim", notOwnedReason(Target{Project: "P", Repo: "r"}, Claim{Kind: "repos"}),
			"not owned by this team in teams.yaml; ask the noergler admin"},
		{"unclaimed project", notOwnedReason(Target{Project: "P"}, Claim{Kind: "none"}),
			"not owned by this team in teams.yaml; ask the noergler admin"},
		{"whole claim", wholeClaimReason(Target{Project: "P", Repo: "r"}),
			"teams.yaml claims all of P for this team: one project webhook covers it, a repo webhook next to it would deliver every event twice"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s =\n%q\nwant\n%q", c.name, c.got, c.want)
		}
	}
}

func TestUpstreamErrorMessage(t *testing.T) {
	inner := transportErr("dial tcp: no route to host")
	cases := []struct {
		err  *UpstreamError
		want string
	}{
		{&UpstreamError{Target: "PROJ", Status: 500, Err: statusErr(500, "boom")},
			"Bitbucket answered HTTP 500 on PROJ"},
		{&UpstreamError{Target: "PROJ/r", Status: 404, Err: statusErr(404, "")},
			"Bitbucket answered HTTP 404 on PROJ/r"},
		{&UpstreamError{Target: "PROJ", Status: 0, Err: inner},
			"Bitbucket unreachable on PROJ: dial tcp: no route to host"},
	}
	for _, c := range cases {
		if got := c.err.Error(); got != c.want {
			t.Errorf("Error() = %q, want %q", got, c.want)
		}
	}
	wrapped := &UpstreamError{Target: "P", Err: inner}
	if !errors.Is(wrapped, inner) {
		t.Error("UpstreamError does not unwrap to its cause")
	}
}
