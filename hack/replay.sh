#!/usr/bin/env bash
# hack/replay.sh: sign a webhook payload with a team's secret and POST it.
#
# The signature is what the webhook route checks first, so this is the only way
# to reach the review pipeline from outside. Both hack/smoke.sh and
# hack/parity.sh call it rather than keeping their own copy of the HMAC.
#
# Usage: replay.sh <payload-file> [event-key]
# Env: PORT (18080), TEAM (platform), SECRET (s), HOST (localhost)
set -euo pipefail

payload=${1:?usage: replay.sh <payload-file> [event-key]}
event=${2:-pr:opened}
port=${PORT:-18080}
team=${TEAM:-platform}
secret=${SECRET:-s}
host=${HOST:-localhost}

# openssl prints "SHA2-256(stdin)= <hex>" or a bare hex depending on version;
# the last field is the digest either way.
sig=$(openssl dgst -sha256 -hmac "$secret" "$payload" | awk '{print $NF}')

curl -s -X POST \
  -H "X-Hub-Signature: sha256=$sig" \
  -H "X-Event-Key: $event" \
  --data-binary "@$payload" \
  "$host:$port/webhook/$team"
