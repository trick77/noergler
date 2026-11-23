package store

// Pipeline is a review's own record of its run, what the run detail page
// shows beyond the run and call rows: where the time went, what the prompt
// was made of, how each file was sent, and what happened to every finding
// the model proposed. The review writes it and the API serves it, both
// through this one struct, stored as JSONB per attempt.
//
// Everything, code included (decided): the page is reached only by its
// link token from the PR's summary comment. Never a secret: pre-existing
// prose with a secret-like token is dropped (inference.HasSecret).
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
	// Compliance is every ticket requirement the model listed and what
	// became of it: shown in the summary when Drop is empty, else left out
	// for that reason (inference.ComplianceDrop).
	Compliance []ComplianceCheck `json:"compliance,omitempty"`
	// HTTP is a list, not a map: a response body is a struct with a fixed
	// order, never map keys.
	HTTP []HTTPCount `json:"http"`
	// PRScope is the whole PR's reviewable diff at this run's commit, for the
	// PR page; nil when the whole diff was not read (an incremental run whose
	// cumulative fetch failed, or a record from before it existed). The run
	// page never shows it.
	PRScope *PRScope `json:"pr_scope,omitempty"`
}

// PRScope is the whole PR's reviewable files and lines, counted as the
// merge-time rollup counts them.
type PRScope struct {
	Added   int `json:"added"`
	Removed int `json:"removed"`
	Files   int `json:"files"`
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
	// Found and Tokens are the root file's.
	Found      bool `json:"found"`
	Tokens     int  `json:"tokens"`
	WarnTokens int  `json:"warn_tokens"`
	// Name is the root file's name, the team's choice; absent on runs from
	// before it was recorded, which read AGENTS.md.
	Name string `json:"name,omitempty"`
	// Empty: the root file is there but blank, so the review ran without
	// its rules (Found is false then); the run page says empty, not missing.
	Empty bool `json:"empty,omitempty"`
	// Nested are the files below the root the prompt carried, in path
	// order; LeftOut the ones that applied but were not carried, and why.
	// Absent on runs from before they were recorded.
	Nested  []AgentsMDFile `json:"nested,omitempty"`
	LeftOut []AgentsMDLeft `json:"left_out,omitempty"`
	// Incomplete is why the lookup did not probe every candidate
	// directory: "probe_cap" or "unreadable" (a failed read ended it);
	// empty when it did.
	Incomplete string `json:"incomplete,omitempty"`
}

// AgentsMDFile is one nested AGENTS.md the prompt carried, with what it
// cost (its heading included).
type AgentsMDFile struct {
	Path   string `json:"path"`
	Tokens int    `json:"tokens"`
}

// AgentsMDLeft is one nested AGENTS.md that applied but was not carried.
// Reason is "token_limit" or "oversize".
type AgentsMDLeft struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
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
	File       string   `json:"file"`
	Line       int      `json:"line"`
	Severity   string   `json:"severity"`
	Headline   string   `json:"headline"`
	Comment    string   `json:"comment,omitempty"`
	Suggestion string   `json:"suggestion,omitempty"`
	Evidence   []string `json:"evidence,omitempty"`
	Decision   string   `json:"decision"`
	Reason     string   `json:"reason,omitempty"`
}

// MovedFinding is a POSTED finding validation moved to its evidence's line.
type MovedFinding struct {
	File string `json:"file"`
	From int    `json:"from"`
	To   int    `json:"to"`
}

// VerdictChange is the model's verdict and the one posted: lowered by drops,
// raised by earlier findings still open.
type VerdictChange struct {
	Model  string `json:"model"`
	Posted string `json:"posted"`
}

// HTTPCount is one upstream's request count for the run.
type HTTPCount struct {
	Upstream string `json:"upstream"`
	Calls    int    `json:"calls"`
}

// ComplianceCheck is one ticket requirement the model listed: the line it
// quoted as proof, and why the summary left it out (empty when shown).
type ComplianceCheck struct {
	Requirement string `json:"requirement"`
	File        string `json:"file,omitempty"`
	Quote       string `json:"quote,omitempty"`
	Drop        string `json:"drop,omitempty"`
}

// PreExistingBug is one bug in unchanged code: where, a headline, the
// model's explanation (capped in length) and the lines it quoted.
type PreExistingBug struct {
	File        string   `json:"file"`
	Line        int      `json:"line"`
	Headline    string   `json:"headline"`
	Explanation string   `json:"explanation"`
	Evidence    []string `json:"evidence,omitempty"`
}
