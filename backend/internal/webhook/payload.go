// Package webhook holds the Bitbucket Server webhook payload types.
//
// Decoding into a plain Go struct accepts anything that is valid JSON, so
// Validate refuses a payload missing a required field at the edge. Without
// it a payload with no pullRequest.author would panic deep inside the review
// path.
package webhook

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Event keys this service acts on.
const (
	EventOpened         = "pr:opened"
	EventFromRefUpdated = "pr:from_ref_updated"
	EventMerged         = "pr:merged"
	EventDeclined       = "pr:declined"
	EventDeleted        = "pr:deleted"
	EventCommentAdded   = "pr:comment:added"
	EventCommentDeleted = "pr:comment:deleted"
)

// Project is the Bitbucket project a repository belongs to.
type Project struct {
	Key string `json:"key"`
}

// Repository identifies one repository.
type Repository struct {
	Slug    string  `json:"slug"`
	Project Project `json:"project"`
}

// Ref is a branch endpoint of the pull request.
//
// LatestCommit and Repository are optional: Bitbucket omits the repository on
// one side in some payload shapes, which is why the project/repo extraction
// tries toRef first and then fromRef.
type Ref struct {
	ID           string      `json:"id"`
	DisplayID    string      `json:"displayId"`
	LatestCommit string      `json:"latestCommit"`
	Repository   *Repository `json:"repository"`
}

// User is a Bitbucket account. Only Name is required.
type User struct {
	Name        string `json:"name"`
	Slug        string `json:"slug"`
	DisplayName string `json:"displayName"`
}

// Participant wraps the user on the author field.
type Participant struct {
	User User `json:"user"`
}

// MergeCommit carries the merge commit SHA on a pr:merged payload.
type MergeCommit struct {
	ID string `json:"id"`
}

// Properties is the optional property bag; only mergeCommit is read.
type Properties struct {
	MergeCommit *MergeCommit `json:"mergeCommit"`
}

// PullRequest is the PR the event concerns.
type PullRequest struct {
	ID          int         `json:"id"`
	Title       string      `json:"title"`
	State       string      `json:"state"`
	FromRef     Ref         `json:"fromRef"`
	ToRef       Ref         `json:"toRef"`
	Author      Participant `json:"author"`
	CreatedDate int64       `json:"createdDate"` // Bitbucket epoch milliseconds
	Properties  *Properties `json:"properties"`
}

// CommentParent is the comment a reply hangs under.
type CommentParent struct {
	ID int `json:"id"`
}

// Comment is the comment on a pr:comment:* event.
type Comment struct {
	ID     int            `json:"id"`
	Text   string         `json:"text"`
	Author User           `json:"author"`
	Parent *CommentParent `json:"parent"`
}

// Payload is one Bitbucket webhook delivery.
type Payload struct {
	EventKey    string      `json:"eventKey"`
	PullRequest PullRequest `json:"pullRequest"`
	Comment     *Comment    `json:"comment"`
	Actor       *User       `json:"actor"`
}

// ErrInvalidPayload is returned when a required field is missing.
var ErrInvalidPayload = errors.New("invalid webhook payload")

// Decode unmarshals and validates a webhook payload.
//
// Unknown fields are allowed: Bitbucket adds fields across versions and a
// strict decode here would refuse deliveries the service can handle. That is
// the opposite of teams.yaml, where an unknown key is an operator typo and
// disables the team.
func Decode(data []byte) (*Payload, error) {
	var p Payload
	if err := json.Unmarshal(data, &p); err != nil {
		// Both wrapped: the sentinel so callers can errors.Is it, and the json
		// error so the reason survives. Dropping either one loses information a
		// caller needs.
		return nil, fmt.Errorf("%w: %w", ErrInvalidPayload, err)
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return &p, nil
}

// Validate reports whether every required field is present: eventKey,
// pullRequest.id, .title, .fromRef (id or displayId), .toRef,
// .author.user.name, and on a comment id and author.name. Everything else is
// optional and decodes to a zero value.
func (p *Payload) Validate() error {
	switch {
	case p.EventKey == "":
		return fmt.Errorf("%w: eventKey is required", ErrInvalidPayload)
	case p.PullRequest.ID == 0:
		return fmt.Errorf("%w: pullRequest.id is required", ErrInvalidPayload)
	case p.PullRequest.Title == "":
		return fmt.Errorf("%w: pullRequest.title is required", ErrInvalidPayload)
	case p.PullRequest.Author.User.Name == "":
		return fmt.Errorf("%w: pullRequest.author.user.name is required", ErrInvalidPayload)
	case p.PullRequest.FromRef.ID == "" && p.PullRequest.FromRef.DisplayID == "":
		return fmt.Errorf("%w: pullRequest.fromRef is required", ErrInvalidPayload)
	case p.PullRequest.ToRef.ID == "" && p.PullRequest.ToRef.DisplayID == "":
		return fmt.Errorf("%w: pullRequest.toRef is required", ErrInvalidPayload)
	}
	if c := p.Comment; c != nil {
		switch {
		case c.ID == 0:
			return fmt.Errorf("%w: comment.id is required", ErrInvalidPayload)
		// comment.text is deliberately not required: a payload with text ""
		// validates. Rejecting it here would answer 400 instead of 200
		// "comment without mention". The route gates on the @mention trigger,
		// so an empty text never reaches the model. Pinned by
		// TestEmptyCommentTextIsAccepted.
		case c.Author.Name == "":
			return fmt.Errorf("%w: comment.author.name is required", ErrInvalidPayload)
		}
	}
	return nil
}

// ProjectRepo extracts the project key and repo slug.
//
// Bitbucket Server nests the repository under fromRef/toRef, and either side
// may carry it, so toRef is tried first and fromRef second.
// Both empty means the payload is unusable for a review.
func (p *Payload) ProjectRepo() (project, repo string) {
	for _, ref := range []Ref{p.PullRequest.ToRef, p.PullRequest.FromRef} {
		if ref.Repository != nil {
			return ref.Repository.Project.Key, ref.Repository.Slug
		}
	}
	return "", ""
}

// MergeCommitSHA pulls the merge commit SHA from a pr:merged payload.
//
// Riptide's pr_completed validator rejects a merged event without one, so a
// missing SHA fails loudly downstream rather than here. An empty string is
// treated as missing: some Bitbucket deployments populate the field as "".
func (p *Payload) MergeCommitSHA() string {
	props := p.PullRequest.Properties
	if props == nil || props.MergeCommit == nil {
		return ""
	}
	return props.MergeCommit.ID
}
