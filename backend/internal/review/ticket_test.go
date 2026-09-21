package review

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/webhook"
)

// The prompt's ticket block had no test at all: formatTicketBlock sat at 0%
// and renderTicketContext at 22%, because no review test ever built a
// Reviewer with a Jira client. These exercise the formatter directly and then
// drive one review end to end through it.

func fullTicket() *jira.Ticket {
	return &jira.Ticket{
		Key:                "PROJ-123",
		Title:              "Add the widget",
		Description:        "A widget is needed.",
		Labels:             []string{"backend", "urgent"},
		AcceptanceCriteria: "AC1: it widgets",
		Subtasks:           []string{"PROJ-124 build it", "PROJ-125 test it"},
		URL:                "https://jira/browse/PROJ-123",
		IssueType:          "Story",
		Status:             "In Progress",
	}
}

func TestFormatTicketBlockRendersEveryField(t *testing.T) {
	got := strings.Join(formatTicketBlock(fullTicket(), "Jira ticket"), "\n")

	want := strings.Join([]string{
		"### Jira ticket: [PROJ-123](https://jira/browse/PROJ-123)",
		"**Title:** Add the widget",
		"**Type:** Story · **Status:** In Progress",
		"**Description:** A widget is needed.",
		"**Labels:** backend, urgent",
		"**Acceptance criteria:** AC1: it widgets",
		"**Subtasks:**",
		"- PROJ-124 build it",
		"- PROJ-125 test it",
	}, "\n")

	if got != want {
		t.Fatalf("ticket block\n got:\n%s\nwant:\n%s", got, want)
	}
}

// Type and Status share one line, so each of the four combinations renders a
// different thing: both, one, the other, or no line at all.
func TestFormatTicketBlockTypeAndStatus(t *testing.T) {
	cases := []struct {
		name              string
		issueType, status string
		want              string // "" means no type/status line
	}{
		{"both", "Story", "In Progress", "**Type:** Story · **Status:** In Progress"},
		{"type only", "Bug", "", "**Type:** Bug"},
		{"status only", "", "Done", "**Status:** Done"},
		{"neither", "", "", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ticket := &jira.Ticket{Key: "PROJ-1", Title: "T", URL: "u", IssueType: tc.issueType, Status: tc.status}
			lines := formatTicketBlock(ticket, "Jira ticket")

			// Line 0 is the heading, line 1 the title; anything more is the
			// type/status line, which must be absent when both are empty.
			if tc.want == "" {
				if len(lines) != 2 {
					t.Fatalf("expected heading and title only, got %q", lines)
				}
				return
			}
			if len(lines) != 3 {
				t.Fatalf("expected a type/status line, got %q", lines)
			}
			if lines[2] != tc.want {
				t.Errorf("type/status line = %q, want %q", lines[2], tc.want)
			}
		})
	}
}

// Every optional field is dropped when empty, so a bare ticket renders two
// lines and nothing else.
func TestFormatTicketBlockOmitsEmptyFields(t *testing.T) {
	ticket := &jira.Ticket{Key: "PROJ-9", Title: "Bare", URL: "https://jira/browse/PROJ-9"}
	got := strings.Join(formatTicketBlock(ticket, "Jira ticket"), "\n")

	want := "### Jira ticket: [PROJ-9](https://jira/browse/PROJ-9)\n**Title:** Bare"
	if got != want {
		t.Fatalf("bare ticket block\n got:\n%s\nwant:\n%s", got, want)
	}
	for _, unwanted := range []string{"**Type:**", "**Status:**", "**Description:**", "**Labels:**", "**Acceptance criteria:**", "**Subtasks:**"} {
		if strings.Contains(got, unwanted) {
			t.Errorf("bare ticket block contains %s", unwanted)
		}
	}
}

func TestRenderTicketContextNilTicketIsEmpty(t *testing.T) {
	if got := renderTicketContext(nil, nil); got != "" {
		t.Fatalf("nil ticket rendered %q, want empty", got)
	}
	// A parent without a ticket is not a sub-task, it is nothing.
	if got := renderTicketContext(nil, fullTicket()); got != "" {
		t.Fatalf("nil ticket with a parent rendered %q, want empty", got)
	}
}

func TestRenderTicketContextWithoutParent(t *testing.T) {
	got := renderTicketContext(fullTicket(), nil)

	if !strings.HasPrefix(got, "### Jira ticket: [PROJ-123]") {
		t.Errorf("lone ticket should use the Jira ticket heading, got:\n%s", got)
	}
	for _, unwanted := range []string{"Parent ticket", "Sub-task"} {
		if strings.Contains(got, unwanted) {
			t.Errorf("lone ticket names %q:\n%s", unwanted, got)
		}
	}
}

// The parent is rendered first and the ticket becomes the sub-task, separated
// by a blank line.
func TestRenderTicketContextWithParent(t *testing.T) {
	parent := &jira.Ticket{Key: "PROJ-100", Title: "The epic", URL: "https://jira/browse/PROJ-100", IssueType: "Epic"}
	got := renderTicketContext(fullTicket(), parent)

	parentAt := strings.Index(got, "### Parent ticket: [PROJ-100](https://jira/browse/PROJ-100)")
	subAt := strings.Index(got, "### Sub-task: [PROJ-123](https://jira/browse/PROJ-123)")
	if parentAt != 0 {
		t.Fatalf("parent block should open the context, got:\n%s", got)
	}
	if subAt < 0 {
		t.Fatalf("no sub-task block:\n%s", got)
	}
	if !strings.Contains(got, "\n\n### Sub-task:") {
		t.Errorf("parent and sub-task should be separated by a blank line, got:\n%s", got)
	}
	if strings.Contains(got, "### Jira ticket:") {
		t.Errorf("a sub-task must not also render the lone-ticket heading:\n%s", got)
	}
}

// jiraHarness is a review harness with a Jira client attached, which no other
// review test builds.
func jiraHarness(t *testing.T, ticket, parent *jira.Ticket) *harness {
	t.Helper()
	return newHarness(t, func(h *harness) {
		h.jr = &fakeJira{ticket: ticket, parent: parent}
	})
}

// jiraPayload is prPayload with a ticket key on the branch name.
func jiraPayload(event string) *webhook.Payload {
	p := prPayload(event)
	p.PullRequest.FromRef.DisplayID = "feature/PROJ-123-the-widget"
	return p
}

// The whole point of the Jira path: the ticket reaches the model's prompt.
func TestReviewWithJiraPutsTicketInPrompt(t *testing.T) {
	h := jiraHarness(t, fullTicket(), nil)

	h.r.ReviewPullRequest(context.Background(), jiraPayload(webhook.EventOpened), false)

	if len(h.jr.Calls) != 1 || h.jr.Calls[0] != "PROJ-123" {
		t.Fatalf("expected one Jira lookup for PROJ-123, got %v", h.jr.Calls)
	}
	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	prompt := h.llm.Reviews[0].Prompt
	for _, want := range []string{
		"### Jira ticket: [PROJ-123](https://jira/browse/PROJ-123)",
		"**Title:** Add the widget",
		"**Acceptance criteria:** AC1: it widgets",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt is missing %q", want)
		}
	}
}

// A sub-task's parent reaches the prompt too, ahead of the sub-task itself.
func TestReviewWithParentTicketPutsBothInPrompt(t *testing.T) {
	parent := &jira.Ticket{Key: "PROJ-100", Title: "The epic", URL: "https://jira/browse/PROJ-100"}
	h := jiraHarness(t, fullTicket(), parent)

	h.r.ReviewPullRequest(context.Background(), jiraPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	prompt := h.llm.Reviews[0].Prompt
	parentAt := strings.Index(prompt, "### Parent ticket: [PROJ-100]")
	subAt := strings.Index(prompt, "### Sub-task: [PROJ-123]")
	switch {
	case parentAt < 0:
		t.Fatalf("prompt has no parent ticket block:\n%s", prompt)
	case subAt < 0:
		t.Fatalf("prompt has no sub-task block:\n%s", prompt)
	case parentAt > subAt:
		t.Errorf("parent block should precede the sub-task, got %d > %d", parentAt, subAt)
	}
}

// The compliance list the model returns reaches the summary a human reads.
func TestReviewWithJiraPutsComplianceInSummary(t *testing.T) {
	h := jiraHarness(t, fullTicket(), nil)
	result := okResultWith()
	result.Review.ComplianceRequirements = []inference.ComplianceRequirement{
		{Requirement: "it widgets", Met: true},
		{Requirement: "it sprockets", Met: false},
	}
	h.llm.review = result

	h.r.ReviewPullRequest(context.Background(), jiraPayload(webhook.EventOpened), false)

	if len(h.bb.Posted) != 1 {
		t.Fatalf("expected one summary comment, got %d", len(h.bb.Posted))
	}
	summary := h.bb.Posted[0].Text
	for _, want := range []string{"[PROJ-123]", "it widgets", "it sprockets"} {
		if !strings.Contains(summary, want) {
			t.Errorf("summary is missing %q, got:\n%s", want, summary)
		}
	}
}

// No key on the branch or title means no lookup at all, even with Jira wired.
func TestReviewWithoutTicketKeySkipsJira(t *testing.T) {
	h := jiraHarness(t, fullTicket(), nil)

	// prPayload's branch is "feature/thing" and its title "Add new feature";
	// neither carries a key.
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.jr.Calls) != 0 {
		t.Errorf("expected no Jira lookup, got %v", h.jr.Calls)
	}
	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	if strings.Contains(h.llm.Reviews[0].Prompt, "### Jira ticket") {
		t.Error("prompt carries a ticket block with no ticket key")
	}
}

// A Jira failure is not a review failure: the key came off a branch name and
// may be noise, so the review proceeds without a ticket block.
func TestReviewContinuesWhenJiraFails(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.jr = &fakeJira{err: errors.New("jira is down")}
	})

	h.r.ReviewPullRequest(context.Background(), jiraPayload(webhook.EventOpened), false)

	if len(h.jr.Calls) != 1 {
		t.Fatalf("expected the lookup to be attempted, got %v", h.jr.Calls)
	}
	if len(h.llm.Reviews) != 1 {
		t.Fatalf("review should proceed without the ticket, got %d LLM calls", len(h.llm.Reviews))
	}
	if strings.Contains(h.llm.Reviews[0].Prompt, "### Jira ticket") {
		t.Error("prompt carries a ticket block after a failed lookup")
	}
	if len(h.bb.Posted) != 1 {
		t.Errorf("expected the summary to be posted anyway, got %d", len(h.bb.Posted))
	}
}
