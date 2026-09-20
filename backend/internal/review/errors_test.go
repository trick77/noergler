package review

import (
	"errors"
	"fmt"

	"github.com/trick77/noergler-go/internal/bitbucket"
)

// Error values the fakes return. They are the real adapter errors wherever
// the pipeline branches on the type, so a test cannot pass against a shape
// the adapters never produce.
var (
	errUpsert = errors.New("upsert failed")
	errPost   = errors.New("bitbucket rejected the comment")

	// The pipeline matches this with errors.Is, as it does for a rebase.
	errIncrementalUnavailable = fmt.Errorf("compare/diff 406: %w", bitbucket.ErrIncrementalDiffUnavailable)

	// Matched with errors.As, so it must be the real type.
	errDiffTooLarge error = &bitbucket.ContentTooLarge{
		What:  "PROJ/my-repo PR diff",
		Limit: 10 * 1024 * 1024,
	}
)
