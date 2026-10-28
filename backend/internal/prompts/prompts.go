// Package prompts holds the model prompts, compiled into the binary: there is
// no file to mount over, and none to miss at startup.
package prompts

import _ "embed"

// Review is the review prompt. Its placeholder order is load-bearing:
// {files} before {cumulative_pr_diff} and {previously_posted_findings}, for
// the prefix cache.
//
//go:embed review.txt
var Review string

// Mention answers an @mention question.
//
//go:embed mention.txt
var Mention string

// FindingReply answers the PR author's reply on a finding's thread.
//
//go:embed finding_reply.txt
var FindingReply string

// AlmanachNote is the fourth turn of the review conversation: the scratch
// note of what the PR taught about its repository, for Almanach. Always
// German, never shown on the PR.
//
//go:embed almanach_note.txt
var AlmanachNote string
