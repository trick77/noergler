package review

import (
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/riptide"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/tokens"
)

// The interfaces are declared consumer-side, so nothing forces the concrete
// adapters to keep satisfying them. These assertions do: a signature change
// in Phase 3's or Phase 5's packages fails the build here rather than at the
// one call site that happens to use it.
var (
	_ BitbucketClient = (*bitbucket.Client)(nil)
	_ JiraClient      = (*jira.Client)(nil)
	_ RiptideEmitter  = (*riptide.Emitter)(nil)
	_ InferenceClient = (*inference.Client)(nil)
	_ Store           = (*store.Store)(nil)
	_ TokenCounter    = (*tokens.Counter)(nil)
)

func TestConcreteAdaptersSatisfyTheInterfaces(_ *testing.T) {
	// The compile-time assertions above are the test; this keeps go vet from
	// flagging a test file with no test.
}
