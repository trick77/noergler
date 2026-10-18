package store

// Pipeline is a review's own record of its run, what the run detail page
// shows beyond the run and call rows: where the time went, what the prompt
// was made of, how each file was sent, and what happened to every finding
// the model proposed. The review writes it and the API serves it, both
// through this one struct, stored as JSONB per attempt.
//
// Headlines only (decided): a finding's comment, suggestion and evidence
// never go in here. The dashboard is unauthenticated. The one exception is
// PreExisting, decided too: never posted, so the page is the only place its
// explanation can be read; prose only, never code.
type Pipeline struct {
	// Commit is the reviewed source commit, for links into the code.
	Commit           string            `json:"commit,omitempty"`
	Stages           []StageTime       `json:"stages"`
	Prompt           PromptParts       `json:"prompt"`
	Files            []FileSent        `json:"files"`
	CrossFile        []string          `json:"cross_file"`
	AgentsMD         AgentsMDFacts     `json:"agents_md"`
	Ticket           string            `json:"ticket,omitempty"`
	ParentTicket     string            `json:"parent_ticket,omitempty"`
	PreviouslyPosted PreviouslyPosted  `json:"previously_posted"`
	Decisions        []FindingDecision `json:"decisions"`
	Moved            []MovedFinding    `json:"moved"`
	Verdict          *VerdictChange    `json:"verdict,omitempty"`
	// PreExisting are bugs the model saw in code the PR did not change,
	// as validation kept them. Never posted to the PR.
	PreExisting []PreExistingBug `json:"pre_existing"`
	// HTTP is a list, not a map: a response body is a struct with a fixed
	// order, never map keys.
	HTTP []HTTPCount `json:"http"`
}

// StageTime is one stage's wall time, in the order the review ran them.
type StageTime struct {
	Name string `json:"name"`
	MS   int64  `json:"ms"`
}

// PromptParts is the prompt's token breakdown plus the whole and the budget
// it was sized against.
type PromptParts struct {
	Template         int `json:"template"`
	Files            int `json:"files"`
	CumulativeDiff   int `json:"cumulative_diff"`
	PreviouslyPosted int `json:"previously_posted"`
	RepoInstructions int `json:"repo_instructions"`
	Ticket           int `json:"ticket"`
	PRDescription    int `json:"pr_description"`
	Total            int `json:"total"`
	Budget           int `json:"budget"`
}

// FileSent is one changed file and how the review treated it. Sent is one
// of full, diff, content_skipped, deleted, renamed, oversized, over_budget,
// not_reviewable. Tokens are the prompt entry's, zero for a file not sent.
type FileSent struct {
	Path    string `json:"path"`
	Sent    string `json:"sent"`
	Added   int    `json:"added"`
	Removed int    `json:"removed"`
	Tokens  int    `json:"tokens"`
}

// AgentsMDFacts is the repo's AGENTS.md as the prompt carried it.
type AgentsMDFacts struct {
	Found      bool `json:"found"`
	Tokens     int  `json:"tokens"`
	WarnTokens int  `json:"warn_tokens"`
}

// PreviouslyPosted is how many earlier findings the prompt showed, of all.
type PreviouslyPosted struct {
	Shown int `json:"shown"`
	Total int `json:"total"`
}

// FindingDecision is a proposed finding that did NOT end up posted, and
// why: dropped by validation (Reason is the DropReason), a repeat of one
// already posted, cut by the comment cap, or refused by Bitbucket. Posted
// findings are the findings table's.
type FindingDecision struct {
	File     string `json:"file"`
	Line     int    `json:"line"`
	Severity string `json:"severity"`
	Headline string `json:"headline"`
	Decision string `json:"decision"`
	Reason   string `json:"reason,omitempty"`
}

// MovedFinding is a POSTED finding validation moved to its evidence's line.
type MovedFinding struct {
	File string `json:"file"`
	From int    `json:"from"`
	To   int    `json:"to"`
}

// VerdictChange is the model's verdict and the one posted after drops.
type VerdictChange struct {
	Model  string `json:"model"`
	Posted string `json:"posted"`
}

// HTTPCount is one upstream's request count for the run.
type HTTPCount struct {
	Upstream string `json:"upstream"`
	Calls    int    `json:"calls"`
}

// PreExistingBug is one bug in unchanged code: where, a headline and the
// model's explanation, capped in length, no code.
type PreExistingBug struct {
	File        string `json:"file"`
	Line        int    `json:"line"`
	Headline    string `json:"headline"`
	Explanation string `json:"explanation"`
}
