package forward

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// nanoPerUSD is llmwire's cost unit, which the store also uses.
const nanoPerUSD = 1_000_000_000

// NullableString renders an empty string as JSON null: a field that is always
// present in a payload and null when there is nothing to say.
func NullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// OrEmpty keeps a list field an array rather than null.
func OrEmpty(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// ReadErrorBody keeps at most limit bytes of a failed response's body for the
// error message and drains the rest, so the connection can be reused.
func ReadErrorBody(r io.Reader, limit int64) string {
	body, _ := io.ReadAll(io.LimitReader(r, limit))
	_, _ = io.Copy(io.Discard, r)
	return strings.TrimSpace(strings.ToValidUTF8(string(body), "�"))
}

// FormatNanoUSD renders nano-USD as a plain decimal string: no exponent, no
// trailing zeros, always at least one digit either side of the point. Money
// crosses a forwarding boundary as text so neither side rounds it through a
// float.
func FormatNanoUSD(nano int64) string {
	sign := ""
	if nano < 0 {
		sign = "-"
		nano = -nano
	}
	whole := nano / nanoPerUSD
	frac := nano % nanoPerUSD
	if frac == 0 {
		return sign + strconv.FormatInt(whole, 10)
	}
	digits := strings.TrimRight(fmt.Sprintf("%09d", frac), "0")
	return sign + strconv.FormatInt(whole, 10) + "." + digits
}
