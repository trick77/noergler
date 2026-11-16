package webhook

import (
	"testing"

	"github.com/trick77/noergler/internal/render"
)

// A reply that tags someone else is theirs, not the bot's (user,
// 2026-11-16); code and addresses tag nobody.
func TestMentionsOther(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"@SJH , damit das funktioniert, muss du aber im service die alten methodennamen noch anbieten.", true},
		{"frag mal @sjh", true},
		{"frag mal @sjh.", true},
		{`@"jan.saner@example.com" schau mal`, true},
		{"@jürg bitte prüfen", true},
		{"(@sjh) siehe oben", true},
		{"@noergler @sjh was meint ihr?", true},
		{"@noergler warum?", false},
		{`@"noergler" warum?`, false},
		{"@NOERGLER warum?", false},
		{"Das ist aber so", false},
		{"schick es an jan@example.com", false},
		{"`@Transactional` fehlt hier nicht", false},
		{"``a `@Override` b`` steht so drin", false},
		{"so:\n```java\n@Test\nvoid x() {}\n```\npasst", false},
		{"so:\n\n    @Test\n    void x() {}\n", false},
		{"@", false},
		{"@ sjh", false},
		{"@.", false},
	}
	for _, c := range cases {
		if got := MentionsOther(c.text, "noergler"); got != c.want {
			t.Errorf("MentionsOther(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// Every mention the bot writes itself is read back as one.
func TestMentionsOtherReadsRenderMention(t *testing.T) {
	for _, user := range []string{"sjh", "jan.saner", "jan saner", "jan.saner@example.com", "jürg"} {
		if !MentionsOther(render.Mention(user)+" schau mal", "noergler") {
			t.Errorf("mention of %q not read", user)
		}
		if MentionsOther(render.Mention(user)+" schau mal", user) {
			t.Errorf("mention of the bot %q read as someone else", user)
		}
	}
}
