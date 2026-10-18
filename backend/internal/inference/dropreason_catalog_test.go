package inference

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// The run page labels a drop reason from ui/src/i18n's dropReason group and
// prints a code it does not know as it came. Every reason this package can
// write needs an entry there, or a German reader sees an English enum.
// Read off the package's own declarations, so a new reason cannot slip past.
func TestDropReasonsAreInTheUICatalog(t *testing.T) {
	if _, err := os.Stat("../../../ui"); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no ui/ next to backend/")
	}
	var declared []string
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			for _, spec := range gd.Specs {
				vs := spec.(*ast.ValueSpec)
				if id, ok := vs.Type.(*ast.Ident); !ok || id.Name != "DropReason" {
					continue
				}
				for _, val := range vs.Values {
					if lit, ok := val.(*ast.BasicLit); ok {
						if v, err := strconv.Unquote(lit.Value); err == nil && v != "" {
							declared = append(declared, v)
						}
					}
				}
			}
		}
	}
	if len(declared) == 0 {
		t.Fatal("no DropReason constants found")
	}
	for _, file := range []string{"en.ts", "de.ts"} {
		src, err := os.ReadFile("../../../ui/src/i18n/" + file)
		if err != nil {
			t.Fatal(err)
		}
		block := regexp.MustCompile(`(?s)\n  dropReason: \{\n(.*?)\n  \},`).FindSubmatch(src)
		if block == nil {
			t.Fatalf("no dropReason block in %s", file)
		}
		var keys []string
		for _, m := range regexp.MustCompile(`(?m)^\s+(\w+):`).FindAllSubmatch(block[1], -1) {
			keys = append(keys, string(m[1]))
		}
		slices.Sort(keys)
		want := slices.Clone(declared)
		slices.Sort(want)
		if !slices.Equal(keys, want) {
			t.Errorf("%s dropReason = %v, the pipeline writes %v", file, keys, want)
		}
	}
}
