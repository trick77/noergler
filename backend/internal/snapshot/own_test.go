package snapshot

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"
)

func TestOwnLine(t *testing.T) {
	for _, tc := range []struct {
		path, line, prev string
		want             bool
	}{
		{"gradle.properties", "version=0.1.0-SNAPSHOT", "", true},
		{"gradle.properties", "version = 0.1.0-SNAPSHOT", "", true},
		{"gradle.properties", "version: 0.1.0-SNAPSHOT", "", true},
		{"gradle.properties", "springBootVersion=4.1.0-SNAPSHOT", "", false},
		{"gradle.properties", "versionLib=4.1.0-SNAPSHOT", "", false},
		{"gradle.properties", "libSuffix=-SNAPSHOT", "", false},
		{"build.gradle", "version = '0.1.0-SNAPSHOT'", "", true},
		{"build.gradle", "version '0.1.0-SNAPSHOT'", "", true},
		{"build.gradle", `project.version = "0.1.0-SNAPSHOT"`, "", true},
		{"build.gradle", "allprojects { version = '0.1.0-SNAPSHOT' }", "", true},
		{"build.gradle", "group = 'x'; version = '0.1.0-SNAPSHOT'", "", true},
		{"build.gradle", "version = '0.1.0-SNAPSHOT'; implementation 'g:a:2.0-SNAPSHOT'", "", false},
		{"build.gradle", "id 'org.x' version '1.0-SNAPSHOT'", "", false},
		{"build.gradle", "implementation 'g:a:1.0-SNAPSHOT'", "", false},
		{"build.gradle.kts", `version = "0.1.0-SNAPSHOT"`, "group = \"x\"", true},
		{"app/build.gradle.kts", `    rootProject.version = "0.1.0-SNAPSHOT"`, "", true},
		{"build.gradle.kts", `id("org.x") version "1.0-SNAPSHOT"`, "", false},
		{"build.gradle.kts", `implementation("g:a:1.0-SNAPSHOT")`, "", false},
		{"settings.gradle.kts", `version("spring", "4.1.0-SNAPSHOT")`, "", false},
		{"build.gradle.kts", `if (version == "1.0-SNAPSHOT") {`, "", false},
		// Kotlin named arguments of a dependency, one per line.
		{"build.gradle.kts", `    version = "2.0-SNAPSHOT",`, `    name = "lib",`, false},
		{"build.gradle.kts", `    version = "2.0-SNAPSHOT"`, `    name = "lib",`, false},
		{"build.gradle.kts", `implementation(group = "g", name = "a", version = "2.0-SNAPSHOT")`, "", false},
		{"build.gradle.kts", `    version = "2.0-SNAPSHOT")`, "", false},
		{".mvn/maven.config", "-Drevision=1.0-SNAPSHOT", "", true},
		{".mvn/maven.config", "-Dchangelist=-SNAPSHOT", "", true},
		{".mvn/maven.config", "-Drevision=1.0-SNAPSHOT -Dlib.version=2.0-SNAPSHOT", "", false},
		{".mvn/maven.config", "-Drevision=1.0-SNAPSHOT -Dlib.version=2.0-SNAPSHOT", "", false},
		{"pom.xml", "<version>1.0-SNAPSHOT</version>", "", false},
		{"gradle/libs.versions.toml", `version = "1.0-SNAPSHOT"`, "", false},
	} {
		if got := OwnLine(tc.path, tc.line, tc.prev, true); got != tc.want {
			t.Errorf("OwnLine(%q, %q, %q) = %v, want %v", tc.path, tc.line, tc.prev, got, tc.want)
		}
	}
}

// The user's case: gradle.properties of a new component, own version only.
func TestAddedOwnGradleVersion(t *testing.T) {
	got := added(`diff --git src://gradle.properties dst://gradle.properties
--- src://gradle.properties
+++ dst://gradle.properties
@@ -0,0 +1,4 @@
+version=0.1.0-SNAPSHOT
+group=ch.example
+springBootVersion=4.1.1
+libVersion=2.0-SNAPSHOT
`)
	if want := []string{"gradle.properties:4 libVersion=2.0-SNAPSHOT"}; !slices.Equal(lines(got), want) {
		t.Fatalf("hits = %q, want %q", lines(got), want)
	}
}

// A Kotlin dependency in named arguments, one per line: a dependency.
func TestAddedKotlinNamedArguments(t *testing.T) {
	got := added(`diff --git src://build.gradle.kts dst://build.gradle.kts
--- src://build.gradle.kts
+++ dst://build.gradle.kts
@@ -1,1 +1,6 @@
 dependencies {
+    implementation(
+        group = "org.outside",
+        name = "lib",
+        version = "2.0-SNAPSHOT",
+    )
`)
	if want := []string{`build.gradle.kts:5         version = "2.0-SNAPSHOT",`}; !slices.Equal(lines(got), want) {
		t.Fatalf("hits = %q, want %q", lines(got), want)
	}
	// A comment between, the last argument without a comma.
	got = added(`diff --git src://build.gradle.kts dst://build.gradle.kts
--- src://build.gradle.kts
+++ dst://build.gradle.kts
@@ -1,1 +1,6 @@
 dependencies {
+    implementation(
+        group = "org.outside", name = "lib",
+        // pinned until 2.0 ships
+        version = "2.0-SNAPSHOT"
+    )
`)
	if want := []string{`build.gradle.kts:5         version = "2.0-SNAPSHOT"`}; !slices.Equal(lines(got), want) {
		t.Fatalf("comment between: hits = %q, want %q", lines(got), want)
	}
	// A hunk starting right at the line mid-file: the line before unknown,
	// never own; at the file's start nothing is before, own.
	for start, want := range map[int]int{7: 1, 1: 0} {
		d := fmt.Sprintf("diff --git src://build.gradle.kts dst://build.gradle.kts\n--- src://build.gradle.kts\n+++ dst://build.gradle.kts\n"+
			"@@ -%d,0 +%d,1 @@\n+version = \"0.1.0-SNAPSHOT\"\n", start, start)
		if got := added(d); len(got) != want {
			t.Errorf("hunk at %d: hits = %q, want %d", start, lines(got), want)
		}
	}
}

func TestGradleVersionKeys(t *testing.T) {
	got := GradleVersionKeys(`plugins { id("x") version "1.0" }
group = "ch.example"
version = projectVersion
allprojects { version = findProperty("appVersion") ?: "0" }
dependencies {
    implementation(
        group = "g",
        version = libVersion
    )
}`)
	for k, want := range map[string]bool{"projectVersion": true, "appVersion": true, "libVersion": false, "group": false} {
		if got[k] != want {
			t.Errorf("key %q = %v, want %v (keys %v)", k, got[k], want, got)
		}
	}
	d := ParseDiff("gradle.properties", "diff --git src://gradle.properties dst://gradle.properties\n--- src://gradle.properties\n+++ dst://gradle.properties\n"+
		"@@ -0,0 +1,2 @@\n+projectVersion=0.1.0-SNAPSHOT\n+libVersion=2.0-SNAPSHOT\n")
	if hits := lines(d.Hits(d.PropertyOwn(got))); !slices.Equal(hits, []string{"gradle.properties:2 libVersion=2.0-SNAPSHOT"}) {
		t.Fatalf("hits = %q", hits)
	}
}

// The own version's bump never exempts a dependency's line in its block.
func TestAddedOwnBumpExemptsNoDependency(t *testing.T) {
	got := added(`diff --git src://build.gradle.kts dst://build.gradle.kts
--- src://build.gradle.kts
+++ dst://build.gradle.kts
@@ -1,2 +1,2 @@
 group = "x"
-version = "1.0-SNAPSHOT"
+implementation("g:a:2.0-SNAPSHOT")
`)
	if want := []string{`build.gradle.kts:2 implementation("g:a:2.0-SNAPSHOT")`}; !slices.Equal(lines(got), want) {
		t.Fatalf("hits = %q, want %q", lines(got), want)
	}
}

const reactorRoot = `<?xml version="1.0" encoding="ISO-8859-1"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <groupId>ch.example</groupId>
  <artifactId>root</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <core.version>1.0-SNAPSHOT</core.version>
    <lib.version>2.0-SNAPSHOT</lib.version>
  </properties>
  <dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>ch.example</groupId>
        <artifactId>core</artifactId>
        <version>1.0-SNAPSHOT</version>
      </dependency>
      <dependency>
        <groupId>org.outside</groupId>
        <artifactId>lib</artifactId>
        <version>
          3.0-SNAPSHOT
        </version>
      </dependency>
    </dependencies>
  </dependencyManagement>
  <description>Zürich&nbsp;build</description>
</project>
`

func ownLines(t *testing.T, content string, reactorParent bool) []int {
	t.Helper()
	f, err := ParsePom(content)
	if err != nil {
		t.Fatal(err)
	}
	return slices.Sorted(maps.Keys(f.OwnLines(reactorParent)))
}

// A declared ISO-8859-1 charset and an HTML entity parse, as in Maven.
func TestPomOwnLinesRoot(t *testing.T) {
	// Own version, the property holding it, the module's managed version.
	if got, want := ownLines(t, reactorRoot, false), []int{5, 7, 15}; !slices.Equal(got, want) {
		t.Fatalf("own lines = %v, want %v", got, want)
	}
}

const module = `<project>
  <parent>
    <groupId>ch.example</groupId>
    <artifactId>root</artifactId>
    <version>1.0-SNAPSHOT</version>
  </parent>
  <artifactId>app</artifactId>
  <dependencies>
    <dependency><groupId>ch.example</groupId><artifactId>core</artifactId><version>1.0-SNAPSHOT</version></dependency>
    <dependency><groupId>org.outside</groupId><artifactId>lib</artifactId><version>2.0-SNAPSHOT</version></dependency>
  </dependencies>
</project>`

// A reactor module inherits the version from its parent: the parent and
// a sibling module are own, an outside dependency is not. A parent from
// outside: nothing is own, its SNAPSHOT counts.
func TestPomOwnLinesModule(t *testing.T) {
	if got, want := ownLines(t, module, true), []int{5, 9}; !slices.Equal(got, want) {
		t.Fatalf("reactor parent: own lines = %v, want %v", got, want)
	}
	if got := ownLines(t, module, false); len(got) != 0 {
		t.Fatalf("outside parent: own lines = %v", got)
	}
}

// A parent from outside counts even with the same version string; the
// CI-friendly properties are own.
func TestPomOwnLinesOutsideParent(t *testing.T) {
	got := ownLines(t, `<project>
  <parent>
    <groupId>org.corp</groupId>
    <artifactId>corp-parent</artifactId>
    <version>7.0-SNAPSHOT</version>
    <relativePath/>
  </parent>
  <artifactId>app</artifactId>
  <version>${revision}${changelist}</version>
  <properties>
    <revision>1.0</revision>
    <changelist>-SNAPSHOT</changelist>
  </properties>
</project>`, false)
	if want := []int{9, 11, 12}; !slices.Equal(got, want) {
		t.Fatalf("own lines = %v, want %v", got, want)
	}
}

func TestPomCoordsAndParentPath(t *testing.T) {
	f, err := ParsePom(module)
	if err != nil {
		t.Fatal(err)
	}
	if want := (Coords{"ch.example", "app", "1.0-SNAPSHOT"}); f.Effective() != want {
		t.Fatalf("effective = %+v", f.Effective())
	}
	for p, want := range map[string]string{"app/pom.xml": "pom.xml", "a/b/pom.xml": "a/pom.xml", "pom.xml": ""} {
		if got := f.ParentPath(p); got != want {
			t.Errorf("ParentPath(%q) = %q, want %q", p, got, want)
		}
	}
	for rel, want := range map[string]string{
		"<relativePath/>":                           "",
		"<relativePath>../parent</relativePath>":    "parent/pom.xml",
		"<relativePath>../bom/x.xml</relativePath>": "bom/x.xml",
		"<relativePath>../../../x</relativePath>":   "",
	} {
		f, err := ParsePom(strings.Replace(module, "</parent>", rel+"</parent>", 1))
		if err != nil {
			t.Fatal(err)
		}
		if got := f.ParentPath("app/pom.xml"); got != want {
			t.Errorf("%s: ParentPath = %q, want %q", rel, got, want)
		}
	}
}

func TestPomBroken(t *testing.T) {
	if _, err := ParsePom("<project><version>1.0-SNAPSHOT</version"); err == nil {
		t.Fatal("broken XML: no error")
	}
}

func pomAdd(content string) string {
	all := strings.Split(strings.TrimSuffix(content, "\n"), "\n")
	return fmt.Sprintf("diff --git src://pom.xml dst://pom.xml\n--- src://pom.xml\n+++ dst://pom.xml\n@@ -0,0 +1,%d @@\n+%s\n",
		len(all), strings.Join(all, "\n+"))
}

// With the whole POM read, a new reactor's own lines bring no hit, the
// outside dependencies do.
func TestAddedInWithPomOwn(t *testing.T) {
	d := ParseDiff("pom.xml", pomAdd(reactorRoot))
	f, err := ParsePom(reactorRoot)
	if err != nil {
		t.Fatal(err)
	}
	got := d.Hits(Own{New: f.OwnLines(false)})
	want := []string{"pom.xml:8     <lib.version>2.0-SNAPSHOT</lib.version>", "pom.xml:21           3.0-SNAPSHOT"}
	if g := hitLines(got); !slices.Equal(g, want) {
		t.Fatalf("hits = %q, want %q", g, want)
	}
	if len(d.Added()) == 0 || !d.Consistent(reactorRoot, false) {
		t.Fatal("candidates or consistency")
	}
	if d.Consistent("\n"+reactorRoot, false) {
		t.Fatal("a shifted file reads consistent")
	}
	// A line the raw read could not decode is skipped, not a mismatch.
	if !d.Consistent(strings.Replace(reactorRoot, "<lib.version>2.0", "<lib.version>�2.0", 1), false) {
		t.Fatal("an undecodable line reads inconsistent")
	}
	// Told by the version strings alone: the own version's lines.
	if got := slices.Sorted(maps.Keys(d.AddedNaming(map[string]bool{"1.0-SNAPSHOT": true}))); !slices.Equal(got, []int{5, 7, 15}) {
		t.Fatalf("named = %v", got)
	}
}

const bumpDiff = `diff --git src://pom.xml dst://pom.xml
--- src://pom.xml
+++ dst://pom.xml
@@ -1,3 +1,4 @@
 <project>
-<version>1.0-SNAPSHOT</version>
+<version>1.1-SNAPSHOT</version>
+<version>2.0-SNAPSHOT</version>
`

// The old side matters only where a removed line may exempt a dependency;
// unknown, a removed line exempts only past the block's own lines.
func TestOldSide(t *testing.T) {
	d := ParseDiff("pom.xml", bumpDiff)
	newOwn := map[int]bool{2: true}
	if !d.OldMatters(newOwn) {
		t.Fatal("own bump beside a dependency: old side does not matter")
	}
	if d.OldMatters(map[int]bool{2: true, 3: true}) {
		t.Fatal("own lines only: old side matters")
	}
	for name, own := range map[string]Own{
		"old known":   {New: newOwn, Old: map[int]bool{2: true}},
		"old unknown": {New: newOwn, OldUnknown: true},
	} {
		if got := lines(d.Hits(own)); !slices.Equal(got, []string{"pom.xml:3 <version>2.0-SNAPSHOT</version>"}) {
			t.Errorf("%s: hits = %q", name, got)
		}
	}
	// A dependency's 1:1 bump, the base unknown: still none (user,
	// 2026-11-15).
	dep := ParseDiff("pom.xml", "diff --git src://pom.xml dst://pom.xml\n--- src://pom.xml\n+++ dst://pom.xml\n@@ -1,2 +1,2 @@\n <project>\n"+
		"-<version>2.0-SNAPSHOT</version>\n+<version>2.1-SNAPSHOT</version>\n")
	if got := dep.Hits(Own{OldUnknown: true}); len(got) != 0 {
		t.Errorf("dependency bump, base unknown: hits = %q", lines(got))
	}
	if !d.Consistent("<project>\n<version>1.0-SNAPSHOT</version>\n", true) ||
		d.Consistent("<project>\n<x/>\n<version>1.0-SNAPSHOT</version>\n", true) {
		t.Fatal("old side consistency")
	}
	if got := d.TakenAsOwn(Own{New: newOwn}); !slices.Equal(got, []int{2}) {
		t.Fatalf("taken as own = %v", got)
	}
}

// hitLines is lines for line numbers past 9.
func hitLines(hs []Hit) []string {
	var out []string
	for _, h := range hs {
		out = append(out, fmt.Sprintf("%s:%d %s", h.Path, h.Line, h.Text))
	}
	return out
}
