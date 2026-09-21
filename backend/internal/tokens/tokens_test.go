package tokens

import (
	"strings"
	"testing"
)

// Counts produced by Python tiktoken.encoding_for_model("gpt-4o") over the same
// inputs. They are literals on purpose: a library upgrade that shifts
// tokenization has to fail here rather than silently reprice every review.
func TestCountMatchesPython(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	cases := []struct {
		name string
		text string
		want int
	}{
		{"ascii", "The quick brown fox jumps over the lazy dog.", 10},
		{"diff_header", "diff --git a/internal/diff/parse.go b/internal/diff/parse.go", 17},
		{"german_umlaut", "Große Straße, außen heißt es kalt: Bär, Tür, Über, Maß.", 18},
		{"repeated_block", strings.Repeat("func handle(w http.ResponseWriter, r *http.Request) {\n\tlog.Println(\"ok\")\n}\n", 10), 190},
		{"empty", "", 0},
		{"hunk_header", "@@ -5,1 +5,1 @@ func main() {", 14},
		{"cjk", "你好世界。こんにちは世界。안녕하세요 세계.", 10},
		{"unicode_ident", "const café = (x) => x", 8},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := c.Count(tc.text); got != tc.want {
				t.Errorf("Count() = %d, want %d (Python tiktoken)", got, tc.want)
			}
		})
	}
}

func TestWarmDoesNotPanic(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c.Warm()
	if got := c.Count("hello"); got == 0 {
		t.Error("Count after Warm returned 0 for non-empty input")
	}
}

// The codec is shared across review goroutines, so concurrent Count must be safe.
func TestCountConcurrent(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	const text = "func main() { fmt.Println(\"hi\") }"
	want := c.Count(text)

	done := make(chan int, 8)
	for i := 0; i < 8; i++ {
		go func() { done <- c.Count(text) }()
	}
	for i := 0; i < 8; i++ {
		if got := <-done; got != want {
			t.Errorf("concurrent Count() = %d, want %d", got, want)
		}
	}
}
