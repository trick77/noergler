package snapshot

import (
	"encoding/xml"
	"errors"
	"io"
	"path"
	"regexp"
	"strings"
)

// The project's own version is a SNAPSHOT on the default branch by design
// (user, 2026-11-16): only a dependency from outside the build blocks a
// merge, never the own version, a reactor module's or what builds it.
//
// By the line alone (OwnLine): gradle.properties' `version` key; a Gradle
// script's `version` statement (Groovy and Kotlin DSL), never a plugin's
// `id(…) version "…"`, a catalog's `version("x", "…")`, a constraint's
// `version { … }` or a named argument (`version = "…",` after `group =
// "…",`); `.mvn/maven.config`'s -Drevision, -Dchangelist and -Dsha1.
// Another file tells the rest: gradle.properties' key the build script
// sets the version from (GradleVersionKeys), a pom.xml read whole
// (ParsePom): its `<version>` looks the same everywhere. A line that
// names another SNAPSHOT beside the own version is never own. RE2's \s is
// ASCII: blanks are [\s\p{Zs}] (INVARIANTS: adapters).

// gradleVersionRE is a Gradle script statement setting the project's
// version to a literal: at the line's start or after `{`/`;`, optionally
// on `project.` or `rootProject.`, then `=` or blanks, then the quoted
// value.
var gradleVersionRE = regexp.MustCompile(`(?:^|[{;])[\s\p{Zs}]*(?:(?:project|rootProject)[\s\p{Zs}]*\.[\s\p{Zs}]*)?version(?:[\s\p{Zs}]*=[\s\p{Zs}]*|[\s\p{Zs}]+)(?:"[^"]*"|'[^']*')`)

// propertiesVersionRE is gradle.properties' `version` key: `=`, `:` or a
// blank ends a key.
var propertiesVersionRE = regexp.MustCompile(`^[\s\p{Zs}]*version[\s\p{Zs}]*[=:\s\p{Zs}]`)

// mavenConfigOwnRE is a maven.config argument setting a CI-friendly
// version part.
var mavenConfigOwnRE = regexp.MustCompile(`-D(?:revision|changelist|sha1)=[^\s\p{Zs}]*`)

// OwnLine reports a build file line that names the project's own version
// by the line alone. prev is the last line before it that is no blank or
// comment, prevKnown whether it is known (a diff's hunk may start right
// at it). A pom.xml's are known only from the whole file (ParsePom), never
// here.
func OwnLine(p, l, prev string, prevKnown bool) bool {
	base := path.Base(p)
	switch {
	case base == "gradle.properties":
		return propertiesVersionRE.MatchString(l)
	case strings.HasSuffix(base, ".gradle") || strings.HasSuffix(base, ".gradle.kts"):
		return prevKnown && gradleOwn(l, prev)
	case base == "maven.config":
		// One line may carry several arguments: own only when nothing but
		// the CI-friendly ones names a SNAPSHOT.
		return mavenConfigOwnRE.MatchString(l) && !Line(mavenConfigOwnRE.ReplaceAllString(l, ""))
	}
	return false
}

// gradleOwn is OwnLine for a Gradle script. A `version = …` that a call
// continues (`,` after it) or that follows an open argument list (the
// line before ends in `(` or `,`) is a named argument of a dependency,
// Kotlin's `implementation(group = …, version = …)`, never the project's.
// An unknown line before counts as one (OwnLine): never own then.
func gradleOwn(l, prev string) bool {
	loc := gradleVersionRE.FindStringIndex(l)
	if loc == nil || namedArgument(l[loc[1]:], prev) {
		return false
	}
	if after := strings.TrimLeft(l[loc[1]:], " \t "); strings.HasPrefix(after, ")") {
		return false
	}
	return !Line(l[:loc[0]] + l[loc[1]:])
}

// namedArgument reports a `version` assignment that is an argument of a
// call: a `,` after its value, or the line before ending in `(` or `,`.
func namedArgument(rest, prev string) bool {
	if strings.HasSuffix(strings.TrimSpace(rest), ",") || strings.HasPrefix(strings.TrimLeft(rest, " \t "), ",") {
		return true
	}
	p := strings.TrimSpace(prev)
	return strings.HasSuffix(p, ",") || strings.HasSuffix(p, "(")
}

// gradleVersionFromRE is a Gradle script statement setting the project's
// version from anything (`version = projectVersion`,
// `version = findProperty("projectVersion")`): the value is group 1.
var gradleVersionFromRE = regexp.MustCompile(`(?:^|[{;])[\s\p{Zs}]*(?:(?:project|rootProject)[\s\p{Zs}]*\.[\s\p{Zs}]*)?version(?:[\s\p{Zs}]*=[\s\p{Zs}]*|[\s\p{Zs}]+)([^={].*)$`)

// keyRE is a name a version statement's value may read a property by.
var keyRE = regexp.MustCompile(`[\pL_][\pL\pN_.\-]*`)

// GradleVersionKeys is the gradle.properties keys a Gradle build script
// sets the project's version from: every name in the value of a
// `version` statement (`projectVersion`, `findProperty("appVersion")`),
// never a named argument's. Blunt: a name that is no key matches none.
func GradleVersionKeys(script string) map[string]bool {
	keys := map[string]bool{}
	var b before
	for _, l := range strings.Split(script, "\n") {
		l = strings.TrimSuffix(l, "\r")
		if m := gradleVersionFromRE.FindStringSubmatchIndex(l); m != nil && !namedArgument(l[m[2]:], b.text) {
			for _, k := range keyRE.FindAllString(l[m[2]:m[3]], -1) {
				keys[k] = true
			}
		}
		b = b.next(l)
	}
	return keys
}

// propertyKey is a gradle.properties line's key: up to `=`, `:` or a
// blank.
func propertyKey(l string) string {
	l = strings.TrimLeft(l, " \t ")
	if i := strings.IndexAny(l, "=: \t "); i >= 0 {
		return l[:i]
	}
	return l
}

// Pom reports a Maven POM, the build file whose own version needs the
// whole file (ParsePom).
func Pom(p string) bool { return path.Base(p) == "pom.xml" }

// Coords is a POM's coordinates, as written: empty when not.
type Coords struct{ GroupID, ArtifactID, Version string }

// PomFile is what a POM tells about its own version.
type PomFile struct {
	// Own is the project's coordinates as written, Parent its parent's.
	Own, Parent Coords
	// RelativePath is the parent's `<relativePath>`; Explicit when given,
	// an empty one included (the parent is looked up outside).
	RelativePath string
	Explicit     bool
	leaves       []pomLeaf
}

// pomLeaf is an element of a POM without child elements.
type pomLeaf struct {
	path     string
	from, to int
	text     string
}

// ParsePom reads a POM. Lenient where Maven is (a declared non-UTF-8
// charset: the body arrives as UTF-8 already; HTML entities), strict
// otherwise: a file that does not parse is an error.
func ParsePom(content string) (*PomFile, error) {
	d := xml.NewDecoder(strings.NewReader(content))
	d.CharsetReader = func(_ string, r io.Reader) (io.Reader, error) { return r, nil }
	d.Strict = false
	d.Entity = xml.HTMLEntity
	var (
		stack []string
		f     PomFile
		text  strings.Builder
		from  int
		leaf  bool // no child element since the last start
	)
	for {
		tok, err := d.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		line, _ := d.InputPos()
		switch t := tok.(type) {
		case xml.StartElement:
			stack = append(stack, t.Name.Local)
			text.Reset()
			from, leaf = line, true
		case xml.CharData:
			text.Write(t)
		case xml.EndElement:
			if len(stack) == 0 {
				return nil, errors.New("pom: end element without a start")
			}
			if leaf {
				l := pomLeaf{path: strings.Join(stack, "/"), from: from, to: line, text: strings.TrimSpace(text.String())}
				f.leaves = append(f.leaves, l)
				f.note(l)
			}
			stack, leaf = stack[:len(stack)-1], false
		}
	}
	if len(stack) != 0 {
		return nil, errors.New("pom: unclosed element")
	}
	return &f, nil
}

func (f *PomFile) note(l pomLeaf) {
	switch l.path {
	case "project/groupId":
		f.Own.GroupID = l.text
	case "project/artifactId":
		f.Own.ArtifactID = l.text
	case "project/version":
		f.Own.Version = l.text
	case "project/parent/groupId":
		f.Parent.GroupID = l.text
	case "project/parent/artifactId":
		f.Parent.ArtifactID = l.text
	case "project/parent/version":
		f.Parent.Version = l.text
	case "project/parent/relativePath":
		f.RelativePath, f.Explicit = l.text, true
	}
}

// Effective is the project's coordinates with what it inherits filled in.
func (f *PomFile) Effective() Coords {
	c := f.Own
	if c.GroupID == "" {
		c.GroupID = f.Parent.GroupID
	}
	if c.Version == "" {
		c.Version = f.Parent.Version
	}
	return c
}

// ParentPath is where in the repo the POM at p looks for its parent
// first: `<relativePath>`, by default `../pom.xml`, a directory meaning
// its pom.xml. "" when there is no parent, an empty `<relativePath/>`
// (looked up outside) or a path out of the repo.
func (f *PomFile) ParentPath(p string) string {
	if f.Parent.ArtifactID == "" || (f.Explicit && f.RelativePath == "") {
		return ""
	}
	rel := f.RelativePath
	if rel == "" {
		rel = "../pom.xml"
	}
	q := path.Join(path.Dir(p), rel)
	if !strings.HasSuffix(q, ".xml") {
		q = path.Join(q, "pom.xml")
	}
	if q == ".." || strings.HasPrefix(q, "../") || strings.HasPrefix(q, "/") {
		return ""
	}
	return q
}

// ParentLines is the lines of the parent's `<version>`, none without one.
func (f *PomFile) ParentLines() map[int]bool {
	out := map[int]bool{}
	for _, l := range f.leaves {
		if l.path == "project/parent/version" {
			for i := l.from; i <= l.to; i++ {
				out[i] = true
			}
		}
	}
	return out
}

// OwnLines is the lines that name the project's own version, by line
// number. reactorParent tells whether the parent is the build's own (its
// POM is in the repo at ParentPath with the same coordinates): only then
// is the parent's version own, and inherited. Own besides: every element
// whose text is the own version (a reactor module's dependency, a
// dependencyManagement entry, a property holding it) and every `revision`,
// `changelist` and `sha1` property (CI-friendly versions). Blunt: an
// outside dependency on the very same version string reads as own.
func (f *PomFile) OwnLines(reactorParent bool) map[int]bool {
	own := f.Own.Version
	if own == "" && reactorParent {
		own = f.Parent.Version
	}
	lines := map[int]bool{}
	for _, l := range f.leaves {
		var isOwn bool
		switch {
		case l.path == "project/parent/version":
			isOwn = reactorParent
		case strings.HasSuffix(l.path, "/properties/revision"), strings.HasSuffix(l.path, "/properties/changelist"),
			strings.HasSuffix(l.path, "/properties/sha1"):
			isOwn = true
		default:
			isOwn = own != "" && l.text == own
		}
		if isOwn {
			for i := l.from; i <= l.to; i++ {
				lines[i] = true
			}
		}
	}
	return lines
}
