// Package api embeds the public OpenAPI contract, so the instance can serve
// it and the contract test can validate real responses against it.
package api

import _ "embed"

//go:embed openapi.yaml
var OpenAPISpec []byte
