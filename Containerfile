# Built with the repo root as context: the image also ships prompts/ and
# builds ui/, both beside backend/. Hence the backend/ prefix on Go paths.

# The dashboard SPA, built first: the Go stage embeds its output, so it has
# to exist on disk before `go build` reads it.
FROM node:26-alpine AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM golang:1.27-alpine AS builder
WORKDIR /build
COPY backend/go.mod backend/go.sum ./
RUN go mod download
COPY backend/ ./
# vite writes into backend/web/dist (ui/vite.config.ts), which in this stage
# is /build/web/dist. Copied AFTER the source, or the COPY above would put
# the tracked .gitkeep back over it.
COPY --from=ui /backend/web/dist ./web/dist
# ship release passes VERSION.
ARG VERSION=dev
# Static binary, no cgo: the tokenizer vocabulary is compiled in, so the
# image needs no runtime besides the CA bundle.
RUN CGO_ENABLED=0 go build -trimpath \
    -ldflags "-s -w -X github.com/trick77/ship/kit/version.Version=${VERSION}" \
    -o /out/noergler ./cmd/noergler

# No shell, no package manager; CA bundle included. Runs as nonroot (65532).
FROM gcr.io/distroless/static-debian13:nonroot
WORKDIR /app
COPY --from=builder /out/noergler /noergler
# Prompt templates ship with the image; the deployment may mount its own
# over /app/prompts (REVIEW_PROMPT_TEMPLATE, REVIEW_MENTION_PROMPT_TEMPLATE,
# REVIEW_FINDING_REPLY_PROMPT_TEMPLATE are relative to the working directory). Read-only to the service.
COPY prompts/ /app/prompts/
# Soft heap ceiling under the pod limit (2 Gi in prod): the runtime collects
# harder as it approaches this instead of letting a 1M-token prompt push RSS
# into the limit. Raise the pod limit together with this.
ENV GOMEMLIMIT=1500MiB
EXPOSE 8080
# The binary probes its own /healthz (SERVER_HOST/SERVER_PORT as serve binds).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/noergler", "-healthcheck"]
ENTRYPOINT ["/noergler"]
CMD ["serve"]
