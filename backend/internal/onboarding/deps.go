package onboarding

import (
	"context"

	"github.com/trick77/noergler-go/internal/bitbucket"
	"github.com/trick77/noergler-go/internal/config"
)

// AdminClient is Bitbucket on the team admin's own token: everything that
// writes, plus the webhook listings that double as the admin-rights proof.
type AdminClient interface {
	ListWebhooks(ctx context.Context, project, repo string) ([]map[string]any, error)
	CreateWebhook(ctx context.Context, project, repo string, body bitbucket.Webhook) (*bitbucket.Webhook, error)
	UpdateWebhook(ctx context.Context, project, repo string, webhookID int, body bitbucket.Webhook) (*bitbucket.Webhook, error)
	DeleteWebhook(ctx context.Context, project, repo string, webhookID int) error
	GrantUserPermission(ctx context.Context, project, repo, username, permission string) error
	ListRepos(ctx context.Context, project string) ([]map[string]any, error)
}

// BotClient is Bitbucket on the bot's own token, used for one thing only:
// proving the bot can read a target.
type BotClient interface {
	GetProject(ctx context.Context, project string) (map[string]any, error)
	GetRepo(ctx context.Context, project, repo string) (map[string]any, error)
	BotUsername() string
}

// ClaimStore is the claims and PR-record side of the store, as the two
// orchestrators in claims.go use it.
type ClaimStore interface {
	AddClaims(ctx context.Context, teamSlug string, scopes []config.ProjectScope, claimedBy string) ([]string, error)
	RemoveClaims(ctx context.Context, teamSlug string, scopes []config.ProjectScope) ([]string, error)
	ListClaims(ctx context.Context, teamSlug string) ([]config.ProjectScope, error)
	PurgeProject(ctx context.Context, teamSlug, projectKey string, repoSlug *string) (int, error)
	CountProjectPRs(ctx context.Context, teamSlug, projectKey string, repoSlug *string) (int, error)
}
