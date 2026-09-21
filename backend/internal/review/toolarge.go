package review

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
)

// maxTooLargeOffenders caps how many files the too-large log line names. The
// question it answers is "what blew the cap", and that is always a handful of
// files; a full listing of a 10 MiB diff would be unreadable.
const maxTooLargeOffenders = 10

// tooLargeOffenders describes the biggest files visible in the truncated head
// of a diff that exceeded its cap, largest first, as "path=bytes".
//
// The final part is kept although it is cut mid-file, marked ">=" because its
// length is a lower bound. Dropping it would report nothing at all in the most
// likely case: a single committed bundle or fixture large enough to be the
// only file in the head.
//
// Returns "" when nothing can be parsed, so the caller can skip the line.
func tooLargeOffenders(head []byte) string {
	if len(head) == 0 {
		return ""
	}
	parts := diff.SplitByFile(string(head))

	type offender struct {
		path  string
		size  int
		short bool // truncated by the cap, so size is a lower bound
	}
	var files []offender
	for i, part := range parts {
		path := diff.ExtractPath(part)
		if path == "" {
			// Preamble before the first `diff --git `, per SplitByFile.
			continue
		}
		files = append(files, offender{path, len(part), i == len(parts)-1})
	}
	if len(files) == 0 {
		return ""
	}
	sort.SliceStable(files, func(i, j int) bool { return files[i].size > files[j].size })

	shown := files
	if len(shown) > maxTooLargeOffenders {
		shown = shown[:maxTooLargeOffenders]
	}
	names := make([]string, 0, len(shown))
	for _, f := range shown {
		sep := "="
		if f.short {
			sep = ">="
		}
		names = append(names, fmt.Sprintf("%s%s%d", f.path, sep, f.size))
	}
	return fmt.Sprintf("%d file(s) in the first %d bytes, largest: %s",
		len(files), len(head), strings.Join(names, " "))
}

// logDiffTooLarge reports a PR diff that exceeded its byte cap: the error
// itself, then the largest files visible in the truncated head.
//
// The second line exists because the cap trips at the socket, before the diff
// is ever split into files, so nothing downstream can say what made the PR too
// big. It is skipped when no path parses, which is the case when the server
// declared a Content-Length and no body was read at all.
func (r *Reviewer) logDiffTooLarge(ctx context.Context, prTag string, err error, tooLarge *bitbucket.ContentTooLarge) {
	r.log.WarnContext(ctx, fmt.Sprintf("%s: %v - skipping review", prTag, err))
	if offenders := tooLargeOffenders(tooLarge.Head); offenders != "" {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: %s", prTag, offenders))
	}
}
