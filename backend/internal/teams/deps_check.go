package teams

import (
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/store"
)

// The real implementations must satisfy the consumer-side interfaces.
// Without these the interfaces drift and the mismatch only shows up in cmd/,
// where there is no test to catch it.
var (
	_ ClaimStore   = (*store.Store)(nil)
	_ review.Store = (*store.Store)(nil)
	_ Reviewer     = (*review.Reviewer)(nil)
)
