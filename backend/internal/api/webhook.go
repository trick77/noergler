package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/trick77/noergler-go/internal/httpapi"
	"github.com/trick77/noergler-go/internal/logging"
	"github.com/trick77/noergler-go/internal/store"
	"github.com/trick77/noergler-go/internal/teams"
	"github.com/trick77/noergler-go/internal/webhook"
)

// maxWebhookBodyBytes bounds an unauthenticated read. A real delivery is well
// under a kilobyte; the headroom is for a comment event carrying a long
// comment body.
const maxWebhookBodyBytes = 1 << 20

// reviewEventKeys are the events that start a review. Python's
// _REVIEW_EVENT_KEYS, exactly.
var reviewEventKeys = map[string]bool{
	webhook.EventOpened:         true,
	webhook.EventFromRefUpdated: true,
}

// accepted and ignored mirror Python's response dicts. They are structs
// rather than maps because encoding/json sorts a map's keys and Python's
// dict order is status, reason, queue.
type accepted struct {
	Status string `json:"status"`
	Reason string `json:"reason,omitempty"`
	PRID   int    `json:"pr_id,omitempty"`
	Queue  string `json:"queue,omitempty"`
}

type ignored struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}

// webhook handles one Bitbucket delivery.
//
// The check order is security-critical and is not the order a Go author
// would write. It follows app/main.py:389 step for step; the comments name
// what each step is load-bearing for. Do not reorder.
func (d Deps) webhook(w http.ResponseWriter, r *http.Request) {
	slug := r.PathValue("team")
	ctx := logging.WithTeam(r.Context(), slug)

	// 1. Team identity comes from the path. Never from the payload.
	rt, ok := d.runtimeFor(ctx, w, slug)
	if !ok {
		return
	}
	team := rt.Team() // one snapshot for every decision below

	// 2. The diagnostics ping is answered before the body is read: Bitbucket
	// sends it to check the endpoint is alive, without a signature.
	if r.Header.Get("X-Event-Key") == "diagnostics:ping" {
		httpapi.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		return
	}

	// 3. Read the raw body. The HMAC covers these exact bytes.
	//
	// Capped, because this read happens BEFORE the signature is checked and
	// the team slug is not a secret: it is the path of the webhook URL every
	// project admin can see. Uncapped, one large POST walks the pod past
	// GOMEMLIMIT. Python read it unbounded; the outbound side has been
	// byte-capped since Phase 3 for the same budget.
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxWebhookBodyBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			d.Log.WarnContext(ctx, "webhook body over the cap", "limit", maxWebhookBodyBytes)
			httpapi.WriteDetail(w, http.StatusRequestEntityTooLarge, "payload too large")
			return
		}
		httpapi.WriteDetail(w, http.StatusBadRequest, "could not read body")
		return
	}

	signature := r.Header.Get("X-Hub-Signature")
	eventKeyHeader := r.Header.Get("X-Event-Key")

	// 4. Bitbucket's "Test connection" omits both the signature and the
	// event key and posts a body with no eventKey in it. Anything else
	// without a signature is refused.
	if signature == "" {
		if eventKeyHeader == "" && !bytes.Contains(body, []byte("eventKey")) {
			d.Log.InfoContext(ctx, "Test connection received (no signature, no event key)")
			httpapi.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
			return
		}
		httpapi.WriteDetail(w, http.StatusUnauthorized, "Missing signature")
		return
	}

	// 5. The signature proves the sender holds THIS team's secret.
	if !verifySignature(body, signature, team.WebhookSecret) {
		httpapi.WriteDetail(w, http.StatusUnauthorized, "Invalid signature")
		return
	}

	// 6. The event key is read off the raw JSON before the payload is
	// validated, because a non-PR event has a different shape entirely and
	// must be ignored with a 200, not refused with a 400.
	var peek struct {
		EventKey string `json:"eventKey"`
	}
	if err := json.Unmarshal(body, &peek); err != nil {
		httpapi.WriteDetail(w, http.StatusBadRequest, "Invalid payload")
		return
	}
	if !strings.HasPrefix(peek.EventKey, "pr:") {
		httpapi.WriteJSON(w, http.StatusOK, ignored{"ignored", "not a PR event: " + peek.EventKey})
		return
	}

	// 7. Now the full payload, with the checks Pydantic made at the edge.
	payload, err := webhook.Decode(body)
	if err != nil {
		d.Log.ErrorContext(ctx, "Failed to parse webhook payload", "error", err)
		httpapi.WriteDetail(w, http.StatusBadRequest, "Invalid payload")
		return
	}
	eventKey := payload.EventKey

	// 8. Bitbucket nests the repository under toRef or fromRef; either side
	// may carry it.
	project, repo := payload.ProjectRepo()
	if project == "" && repo == "" {
		d.Log.ErrorContext(ctx, "event missing repository info",
			"event", eventKey, "pr_id", payload.PullRequest.ID)
		httpapi.WriteJSON(w, http.StatusOK, ignored{"ignored", "missing repository"})
		return
	}

	// 9. The ownership check. The signature proves the sender holds the
	// team's secret; it does NOT prove the PR is the team's. Without this a
	// team could sign a payload naming another team's repo and have it
	// reviewed on its own key.
	if !team.Owns(project, repo) {
		d.Log.WarnContext(ctx, "webhook rejected: repository is not owned by this team",
			"project", project, "repo", repo)
		httpapi.WriteDetail(w, http.StatusForbidden,
			fmt.Sprintf("repository %s/%s is not owned by team %s", project, repo, slug))
		return
	}

	// 10. A project webhook delivers for every repo in it; exclude_repos
	// carves repos out of that for everything that would START a review.
	// Lifecycle events still pass: a PR reviewed before the pattern was set
	// must still be marked merged and get its cost rollup.
	if !team.ReviewsRepo(project, repo) &&
		(reviewEventKeys[eventKey] || eventKey == webhook.EventCommentAdded) {
		d.Log.InfoContext(ctx, "webhook ignored: repo matches exclude_repos",
			"project", project, "repo", repo)
		httpapi.WriteJSON(w, http.StatusOK, ignored{"ignored", "repo excluded by the team's exclude_repos"})
		return
	}

	// 11. Dispatch.
	d.dispatch(ctx, w, rt, slug, project, repo, payload)
}

// dispatch routes an authenticated, owned PR event.
//
// Merge and decline rollups, mention answers and the two deletions all fetch
// a diff or touch the DB, so they run on the review worker rather than in
// the request: one diff and prompt set in memory at a time.
//
// Divergence from Python, pinned in AGENTS.md: pr:deleted and
// pr:comment:deleted run as FastAPI background tasks there and on the queue
// here. Their response bodies keep Python's shape and carry no queue key.
func (d Deps) dispatch(ctx context.Context, w http.ResponseWriter, rt *teams.Runtime, slug, project, repo string, p *webhook.Payload) {
	prTag := fmt.Sprintf("%s/%s#%d", project, repo, p.PullRequest.ID)
	rv := rt.Reviewer

	switch p.EventKey {
	case webhook.EventMerged:
		d.Queue.SubmitJob(prTag, slug, func(ctx context.Context) { rv.HandlePRMerged(ctx, p) })
		httpapi.WriteJSON(w, http.StatusOK, accepted{Status: "accepted", Reason: "merged-rollup", Queue: "queued"})

	case webhook.EventDeclined:
		d.Queue.SubmitJob(prTag, slug, func(ctx context.Context) { rv.HandlePRDeclined(ctx, p) })
		httpapi.WriteJSON(w, http.StatusOK, accepted{Status: "accepted", Reason: "declined-rollup", Queue: "queued"})

	case webhook.EventDeleted:
		d.Queue.SubmitJob(prTag, slug, func(ctx context.Context) { rv.HandlePRDeleted(ctx, p) })
		httpapi.WriteJSON(w, http.StatusOK, accepted{Status: "accepted", Reason: "deleted-purge"})

	case webhook.EventCommentDeleted:
		d.Queue.SubmitJob(prTag, slug, func(ctx context.Context) { rv.HandleCommentDeleted(ctx, p) })
		httpapi.WriteJSON(w, http.StatusOK, accepted{Status: "accepted", Reason: "comment-deleted"})

	case webhook.EventCommentAdded:
		// The mention gate. HandleMention does NOT check that the comment
		// names the bot: without this every comment on every PR would become
		// an inference call.
		var text string
		var commentID int
		if p.Comment != nil {
			text, commentID = p.Comment.Text, p.Comment.ID
		}
		d.Log.InfoContext(ctx, "Comment event", "comment_id", commentID)
		trigger := "@" + d.BotUsername
		if !strings.Contains(strings.ToLower(text), strings.ToLower(trigger)) {
			httpapi.WriteJSON(w, http.StatusOK, ignored{"ignored", "comment without mention"})
			return
		}
		d.Queue.SubmitJob(prTag, slug, func(ctx context.Context) { rv.HandleMention(ctx, p) })
		httpapi.WriteJSON(w, http.StatusOK, accepted{Status: "accepted", Reason: "mention", Queue: "queued"})

	default:
		if !reviewEventKeys[p.EventKey] {
			// The signal that a Bitbucket hook is configured with events we
			// ignore. Warning level on purpose.
			d.Log.WarnContext(ctx, fmt.Sprintf("Unhandled event %q; check your Bitbucket webhook configuration", p.EventKey))
			httpapi.WriteJSON(w, http.StatusOK, ignored{"ignored", "unhandled event: " + p.EventKey})
			return
		}
		key := store.PRKey{Project: project, Repo: repo, PRID: p.PullRequest.ID}
		outcome := d.Queue.Submit(key, p, slug)
		httpapi.WriteJSON(w, http.StatusOK, accepted{Status: "accepted", PRID: p.PullRequest.ID, Queue: outcome})
	}
}
