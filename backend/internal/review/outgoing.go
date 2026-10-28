package review

import (
	"context"

	"github.com/trick77/noergler/internal/render"
)

// outgoing is the Bitbucket client with what every comment the pipeline
// writes needs on its way out, in one place so no post can skip it: no em
// dash (render.NoEmDash), and a reply hard-wrapped like the summary
// (render.WrapProse; user, 2026-10-28). Summaries, inline comments and
// banners are wrapped where they are rendered, field by field.
type outgoing struct{ BitbucketClient }

// withOutgoing wraps c; nil stays nil, the Reviewer that never reaches
// Bitbucket.
func withOutgoing(c BitbucketClient) BitbucketClient {
	if c == nil {
		return nil
	}
	return outgoing{c}
}

func (c outgoing) PostInlineComment(ctx context.Context, project, repo string, prID int, file string, line int, body string) (int, error) {
	return c.BitbucketClient.PostInlineComment(ctx, project, repo, prID, file, line, render.NoEmDash(body))
}

func (c outgoing) PostPRComment(ctx context.Context, project, repo string, prID int, text string) (id, version int, err error) {
	return c.BitbucketClient.PostPRComment(ctx, project, repo, prID, render.NoEmDash(text))
}

func (c outgoing) ReplyToComment(ctx context.Context, project, repo string, prID, parentCommentID int, text string) (int, error) {
	return c.BitbucketClient.ReplyToComment(ctx, project, repo, prID, parentCommentID, render.NoEmDash(render.WrapProse(text)))
}

func (c outgoing) UpdatePRComment(ctx context.Context, project, repo string, prID, commentID, version int, text string) (int, error) {
	return c.BitbucketClient.UpdatePRComment(ctx, project, repo, prID, commentID, version, render.NoEmDash(text))
}
