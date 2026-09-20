// Package buildinfo carries the version stamped into the binary.
package buildinfo

import "os"

// Set at link time:
// -ldflags "-X github.com/trick77/noergler-go/internal/buildinfo.version=1.2.3".
var (
	version string
	commit  string
)

// Version is the release the binary was built as. The ldflags value wins;
// NOERGLER_VERSION is the fallback the Python image used (the infra repo may
// still inject it); "dev" when neither is set.
func Version() string {
	if version != "" {
		return version
	}
	if v := os.Getenv("NOERGLER_VERSION"); v != "" {
		return v
	}
	return "dev"
}

// Commit is the short git hash, empty when not stamped.
func Commit() string { return commit }
