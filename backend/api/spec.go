// Package api embeds the public OpenAPI contract, so the instance can serve
// it and the contract test can validate real responses against it.
package api

import _ "embed"

// OpenAPISpec is the contract as written, served at /api/openapi.yaml and
// validated against the real handlers by the contract test.
//
//go:embed openapi.yaml
var OpenAPISpec []byte
