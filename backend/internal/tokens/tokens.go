// Package tokens counts prompt tokens with the o200k_base encoding, which is
// what gpt-4o resolves to. There is no cl100k_base fallback: gpt-4o always
// resolves, so one would be unreachable.
package tokens

import (
	"fmt"

	"github.com/tiktoken-go/tokenizer"
)

// BytesPerTokenCeiling bounds tokens by byte length.
// The cumulative PR diff is dropped by byte length
// before anything is tokenized, so no encoder runs over a diff that cannot fit.
const BytesPerTokenCeiling = 8

// Counter wraps one o200k_base codec. Build it once at boot and share it; the
// codec is safe for concurrent use and the vocabulary is the expensive part.
type Counter struct {
	codec tokenizer.Codec
}

// New loads the o200k_base vocabulary.
func New() (*Counter, error) {
	codec, err := tokenizer.Get(tokenizer.O200kBase)
	if err != nil {
		return nil, fmt.Errorf("load o200k_base: %w", err)
	}
	return &Counter{codec: codec}, nil
}

// Count returns the number of tokens in text.
func (c *Counter) Count(text string) int {
	ids, _, err := c.codec.Encode(text)
	if err != nil {
		// The o200k_base codec encodes any string: it falls back to byte-level
		// tokens rather than rejecting input. Treat a failure as empty instead
		// of failing a review over a token count.
		return 0
	}
	return len(ids)
}

// Warm encodes a short string so the vocabulary is resident before the first
// review rather than during it, keeping idle RSS the real baseline.
func (c *Counter) Warm() {
	c.Count("")
}
