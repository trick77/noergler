// Package lang is the language noergler writes its prose in. Prompts, logs,
// JSON keys and enums stay English whatever it is; only text a human reads
// on Bitbucket follows it.
package lang

import "strings"

// Lang is a supported output language.
type Lang string

const (
	// DE is the default: Swiss Standard German (ss, never ß).
	DE Lang = "de"
	// EN is the only alternative.
	EN Lang = "en"
)

// Default is the language when neither the user nor the team set one.
const Default = DE

// Parse reads a configured or stored value. ok is false for anything but
// de/en, case-insensitively.
func Parse(s string) (Lang, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "de":
		return DE, true
	case "en":
		return EN, true
	}
	return "", false
}

// Or is l, or def when l is unset.
func (l Lang) Or(def Lang) Lang {
	if l == "" {
		return def
	}
	return l
}

// Pick returns de or en by l; anything but EN is German.
func (l Lang) Pick(de, en string) string {
	if l == EN {
		return en
	}
	return de
}
