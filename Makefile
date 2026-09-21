# The Go module lives in backend/; hack/, ui/ and docs/ are at the root.
VERSION ?= dev

.PHONY: fe-build fe-test fe-coverage build test backend-coverage coverage

# The find, rather than emptyOutDir: vite runs with emptyOutDir false so the
# tracked dist/.gitkeep survives (//go:embed all:dist needs a non-empty
# directory in a fresh clone), which means stale hashed assets from an
# earlier build would otherwise accumulate in the binary.
fe-build:
	find backend/web/dist -mindepth 1 ! -name '.gitkeep' -delete
	cd ui && npm ci && npm run build

# npm run test does not typecheck on its own; npm run build is `tsc -b &&
# vite build`, which is where CI gets its typecheck from.
fe-test:
	cd ui && npx tsc -b && npm run test -- --run

fe-coverage:
	cd ui && npm run test -- --run --coverage
	./hack/coverage-gate.sh ui

build: fe-build
	cd backend && CGO_ENABLED=0 go build \
		-ldflags="-s -w" \
		-o ../bin/noergler ./cmd/noergler

test:
	cd backend && go test -race ./...

backend-coverage:
	mkdir -p coverage
	cd backend && go test -race -covermode=atomic -coverpkg=./... \
		-coverprofile=../coverage/backend.out -count=1 ./...
	cd backend && go run github.com/boumenot/gocover-cobertura@v1.5.0 \
		< ../coverage/backend.out > ../coverage/backend.xml
	./hack/coverage-gate.sh backend

coverage: backend-coverage fe-coverage
