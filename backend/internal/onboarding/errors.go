package onboarding

import "fmt"

// UnknownTarget indicates that a requested target is not in the team's teams.yaml block.
// The HTTP layer answers 400 with Msg.
type UnknownTarget struct{ Msg string }

func (e *UnknownTarget) Error() string { return e.Msg }

// NoClaim indicates that a whole-project remove names a project the team holds nothing on.
// The HTTP layer answers 400 with Msg.
type NoClaim struct{ Msg string }

func (e *NoClaim) Error() string { return e.Msg }

// ForeignHook indicates that a same-named hook points at another noergler instance.
// It is never rewritten, never pruned, never deleted; only reported.
type ForeignHook struct{ Msg string }

func (e *ForeignHook) Error() string { return e.Msg }

// UpstreamError is Bitbucket failing the admin-rights proof for a reason that
// is not "you are not an admin": it aborts the whole request rather than one
// target. The HTTP layer answers 502.
//
// Status is the HTTP status Bitbucket answered, or 0 for a transport failure.
type UpstreamError struct {
	Target string
	Status int
	Err    error
}

func (e *UpstreamError) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("Bitbucket answered HTTP %d on %s", e.Status, e.Target)
	}
	return fmt.Sprintf("Bitbucket unreachable on %s: %v", e.Target, e.Err)
}

func (e *UpstreamError) Unwrap() error { return e.Err }
