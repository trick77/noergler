package forward

import (
	"fmt"
	"strconv"
	"strings"
)

// nanoPerUSD is llmwire's cost unit, which the store also uses.
const nanoPerUSD = 1_000_000_000

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
