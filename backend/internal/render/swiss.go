package render

import "github.com/trick77/noergler/internal/lang"

// SwissSS is lang.SwissSS: Swiss spelling on prose, code left alone. The
// rule lives in lang so inference can apply it to an Almanach note without
// importing this package; render keeps the name its callers use.
func SwissSS(s string) string { return lang.SwissSS(s) }

// OpenFence is lang.OpenFence: the fence a text opens and leaves unclosed.
func OpenFence(s string) string { return lang.OpenFence(s) }
