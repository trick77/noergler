package snapshot

import (
	"slices"
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

func TestLine(t *testing.T) {
	for l, want := range map[string]bool{
		"<version>1.1-SNAPSHOT</version>":                                   true,
		"<version>1.4.0-snapshot</version>":                                 true,
		"implementation 'ch.x:operaton-intranet:1.0.9-SNAPSHOT:api'":        true,
		"version = 1.0.0.BUILD-SNAPSHOT":                                    true,
		"version=2.0-SNAPSHOT\n":                                            true,
		"<version>2.1-20251110.081500-7</version>":                          true,
		"<version>1.0.0-SNAPSHOT-jdk17</version>":                           true,
		"image: registry/app:1.4.0-SNAPSHOT-a1b2c3d":                        true,
		"<version>${revision}-SNAPSHOT</version>":                           true,
		"<version>1.0-SNAPSHOTä</version>":                                  false,
		"<version>1.0.0-beta-SNAPSHOT</version>":                            true,
		"<changelist>-SNAPSHOT</changelist>":                                true,
		"version=1.2.0-my-feature-SNAPSHOT":                                 true,
		`if (version.endsWith("-SNAPSHOT")) {`:                              false,
		"version = version.replace('-SNAPSHOT', '')":                        false,
		"<!-- drop -SNAPSHOT before release -->":                            false,
		"suffix=-SNAPSHOT":                                                  true,
		"<id>LIBS-SNAPSHOTS</id>":                                           false,
		"<id>libs-snapshot</id>":                                            false,
		"url \"https://art.example/artifactory/libs-snapshot\"":             false,
		"<id>nexus-snapshots</id>":                                          false,
		"url 'https://nexus.example/repository/maven-snapshot/'":            false,
		"maven { url = uri(\"https://repo.example/libs-snapshot-local\") }": false,
		"case SNAPSHOT:":                                                    false,
		"takeSnapshot()":                                                    false,
		"<version>1.0.9</version>":                                          false,
	} {
		if got := Line(l); got != want {
			t.Errorf("Line(%q) = %v, want %v", l, got, want)
		}
	}
}

func TestBuildFile(t *testing.T) {
	for p, want := range map[string]bool{
		"pom.xml": true, "lib/pom.xml": true, "build.gradle": true, "app/build.gradle.kts": true,
		"settings.gradle": true, "gradle.properties": true, "gradle/libs.versions.toml": true,
		"gradle/dependencies.gradle": true, ".mvn/maven.config": true, "conf/maven.config": false,
		"buildSrc/src/main/kotlin/Versions.kt": false,
		"values.yaml":                          false, "src/main/java/Pom.java": false, "package.json": false, "pom.xml.orig": false,
	} {
		if got := BuildFile(p); got != want {
			t.Errorf("BuildFile(%q) = %v, want %v", p, got, want)
		}
	}
}

// added is every build file's hits in a whole diff, no Own known.
func added(raw string) []Hit {
	var hits []Hit
	for _, fd := range diff.SplitByFile(raw) {
		if p := diff.ExtractPath(fd); p != "" && BuildFile(p) {
			hits = append(hits, ParseDiff(p, fd).Hits(Own{})...)
		}
	}
	return hits
}

func lines(hs []Hit) []string {
	var out []string
	for _, h := range hs {
		out = append(out, h.Path+":"+string(rune('0'+h.Line))+" "+h.Text)
	}
	return out
}

func TestAdded(t *testing.T) {
	for _, tc := range []struct {
		name, diff string
		want       []string
	}{
		{"added", `diff --git src://build.gradle dst://build.gradle
--- src://build.gradle
+++ dst://build.gradle
@@ -1,2 +1,2 @@
 a
-implementation 'x:lib:1.0'
+implementation 'x:lib:1.1-SNAPSHOT'
`, []string{"build.gradle:2 implementation 'x:lib:1.1-SNAPSHOT'"}},
		{"bumped snapshot", `diff --git src://pom.xml dst://pom.xml
--- src://pom.xml
+++ dst://pom.xml
@@ -1,3 +1,3 @@
 <a>
-<version>1.0-SNAPSHOT</version>
+<version>1.1-SNAPSHOT</version>
 </a>
`, nil},
		{"bumped plus a new one", `diff --git src://pom.xml dst://pom.xml
--- src://pom.xml
+++ dst://pom.xml
@@ -1,2 +1,3 @@
 <a>
-<version>1.0-SNAPSHOT</version>
+<version>1.1-SNAPSHOT</version>
+<version>2.0-SNAPSHOT</version>
`, []string{"pom.xml:3 <version>2.0-SNAPSHOT</version>"}},
		{"bumped on the last line", `diff --git src://gradle.properties dst://gradle.properties
--- src://gradle.properties
+++ dst://gradle.properties
@@ -1,2 +1,2 @@
 group=x
-libVersion=1.0-SNAPSHOT
\ No newline at end of file
+libVersion=1.1-SNAPSHOT
\ No newline at end of file
`, nil},
		{"snapshot in context only", `diff --git src://pom.xml dst://pom.xml
--- src://pom.xml
+++ dst://pom.xml
@@ -1,3 +1,3 @@
 <version>1.0-SNAPSHOT</version>
-<a>1</a>
+<a>2</a>
`, nil},
		{"not a build file", `diff --git src://values.yaml dst://values.yaml
--- src://values.yaml
+++ dst://values.yaml
@@ -1 +1 @@
-tag: 1.0
+tag: 1.1-SNAPSHOT
`, nil},
		{"header names a snapshot", `diff --git src://lib-SNAPSHOT/pom.xml dst://lib-SNAPSHOT/pom.xml
--- src://lib-SNAPSHOT/pom.xml
+++ dst://lib-SNAPSHOT/pom.xml
@@ -1 +1 @@
-<a>1</a>
+<a>2</a>
`, nil},
		{"deleted file", `diff --git src://pom.xml dst://pom.xml
--- src://pom.xml
+++ /dev/null
@@ -1 +0,0 @@
-<version>1.0-SNAPSHOT</version>
`, nil},
	} {
		if got := lines(added(tc.diff)); !slices.Equal(got, tc.want) {
			t.Errorf("%s: Added = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestKeyIgnoresWhitespaceAndLine(t *testing.T) {
	a := Hit{Path: "pom.xml", Line: 3, Text: "  <version>1-SNAPSHOT</version>"}
	b := Hit{Path: "pom.xml", Line: 9, Text: "\t<version>1-SNAPSHOT</version>"}
	c := Hit{Path: "lib/pom.xml", Line: 3, Text: a.Text}
	if a.Key() != b.Key() || a.Key() == c.Key() {
		t.Fatalf("keys: %q %q %q", a.Key(), b.Key(), c.Key())
	}
}

// A per-file response that carries another file too: only the asked one's
// part counts.
func TestFileHits(t *testing.T) {
	raw := `diff --git src://a/pom.xml dst://a/pom.xml
--- src://a/pom.xml
+++ dst://a/pom.xml
@@ -1 +1,2 @@
 <a>
+<version>1.0-SNAPSHOT</version>
diff --git src://b/pom.xml dst://b/pom.xml
--- src://b/pom.xml
+++ dst://b/pom.xml
@@ -1 +1,2 @@
 <b>
+<version>2.0-SNAPSHOT</version>
`
	fileHits := func(p, raw string) []Hit { return ParseDiff(p, FilePart(p, raw)).Hits(Own{}) }
	if got := fileHits("b/pom.xml", raw); len(got) != 1 || got[0].Path != "b/pom.xml" || got[0].Text != "<version>2.0-SNAPSHOT</version>" {
		t.Fatalf("hits = %+v", got)
	}
	if got := fileHits("c/pom.xml", raw); len(got) != 0 {
		t.Fatalf("unknown path: hits = %+v", got)
	}
	single := "@@ -1 +1,2 @@\n <a>\n+<version>1.0-SNAPSHOT</version>\n"
	if got := fileHits("pom.xml", single); len(got) != 1 || got[0].Line != 2 {
		t.Fatalf("headerless: hits = %+v", got)
	}
}

func TestMentions(t *testing.T) {
	for l, want := range map[string]bool{
		"<id>nexus-snapshots</id>": true, `endsWith("-SNAPSHOT")`: true, "v=1.0-20251110.081500-7": true,
		"takeSnapshot()": false, "case SNAPSHOT:": false,
	} {
		if got := Mentions(l); got != want {
			t.Errorf("Mentions(%q) = %v, want %v", l, got, want)
		}
	}
}

// Two dependencies on the same version line are two hits, two keys.
func TestAddedNumbersEqualLines(t *testing.T) {
	hits := added(`diff --git src://pom.xml dst://pom.xml
--- src://pom.xml
+++ dst://pom.xml
@@ -1,1 +1,7 @@
 <a>
+<artifactId>a</artifactId>
+<version>2.0-SNAPSHOT</version>
+<artifactId>b</artifactId>
+<version>2.0-SNAPSHOT</version>
+<artifactId>c</artifactId>
+<version>3.0-SNAPSHOT</version>
`)
	if len(hits) != 3 || hits[0].Seq != 1 || hits[1].Seq != 2 || hits[2].Seq != 1 || hits[0].Key() == hits[1].Key() {
		t.Fatalf("hits = %+v", hits)
	}
}
