# Noergler OpenShift Deployment

## Prerequisites

- `oc` CLI installed and logged in to the cluster
- A project/namespace to deploy into
- Access to a container registry (replace `registry.example.com/noergler` below with your actual registry and namespace)

## 1. Create project

```bash
oc new-project noergler
```

## 2. Create the secret

```bash
oc create secret generic noergler \
  --from-literal=BITBUCKET_TOKEN=<service-account-token> \
  --from-literal=JIRA_TOKEN=<jira-token> \
  --from-literal=DATABASE_URL=<postgresql-url> \
  --from-literal=TEAM_PLATFORM_WEBHOOK_SECRET=<hex-secret> \
  --from-literal=TEAM_PLATFORM_OPENAI_API_KEY=<team-gateway-key>
```

One `TEAM_<SLUG>_WEBHOOK_SECRET` (generate with `openssl rand -hex 32`) and
`TEAM_<SLUG>_OPENAI_API_KEY` per team listed in the `noergler-teams` ConfigMap
(`openshift/configmap.yaml`), plus `TEAM_<SLUG>_RIPTIDE_TOKEN` for a team with a
`riptide:` block.

## 3. Apply manifests

```bash
oc apply -f openshift/
```

## 4. Build and push the image

From the repository root, build the container image and push it to your registry:

```bash
podman build -t registry.example.com/noergler/noergler:latest -f Containerfile .
podman push registry.example.com/noergler/noergler:latest
```

Then restart the deployment to pick up the new image:

```bash
oc rollout restart deploy/noergler
```

## 5. Verify

```bash
oc get pods
oc logs deploy/noergler
curl https://noergler.example.com/health
```

Expected health response: `{"status": "ok", "teams": {"enabled": ["platform"], "disabled": []}}`.
`/ready` answers 503 while `enabled` is empty; the startup log says why
(`team_disabled team=<slug> reason=...`).

## 6. Configure Bitbucket webhook

In Bitbucket Server, add a webhook pointing to:

```
https://noergler.example.com/webhook/<team>
```

with the team's `TEAM_<SLUG>_WEBHOOK_SECRET` as the secret, or run
`python -m scripts.onboard_repo` from the app repo. Create the route manually
before configuring the webhook.

## Rebuilding

After code changes, build and push the updated image:

```bash
podman build -t registry.example.com/noergler/noergler:latest -f Containerfile .
podman push registry.example.com/noergler/noergler:latest
oc rollout restart deploy/noergler
```
