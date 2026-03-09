# Nitpick OpenShift Deployment

## Prerequisites

- `oc` CLI installed and logged in to the cluster
- A project/namespace to deploy into
- Access to a container registry (replace `registry.example.com/nitpick` below with your actual registry and namespace)

## 1. Create project

```bash
oc new-project nitpick
```

## 2. Create the secret

```bash
oc create secret generic nitpick \
  --from-literal=BITBUCKET_URL=https://bitbucket.company.com \
  --from-literal=BITBUCKET_TOKEN=<your-token> \
  --from-literal=BITBUCKET_WEBHOOK_SECRET=<your-secret> \
  --from-literal=GITHUB_TOKEN=<your-token> \
  --from-literal=REVIEW_ALLOWED_AUTHORS=user1,user2
```

## 3. Apply manifests

```bash
oc apply -f openshift/
```

## 4. Build and push the image

From the repository root, build the container image and push it to your registry:

```bash
podman build -t registry.example.com/nitpick/nitpick:latest -f Containerfile .
podman push registry.example.com/nitpick/nitpick:latest
```

Then restart the deployment to pick up the new image:

```bash
oc rollout restart deploy/nitpick
```

## 5. Verify

```bash
oc get pods
oc logs deploy/nitpick
curl https://nitpick.example.com/health
```

Expected health response: `{"status": "ok"}`

## 6. Configure Bitbucket webhook

In Bitbucket Server, add a webhook pointing to:

```
https://nitpick.example.com/webhook
```

Create the route manually before configuring the webhook.

## Rebuilding

After code changes, build and push the updated image:

```bash
podman build -t registry.example.com/nitpick/nitpick:latest -f Containerfile .
podman push registry.example.com/nitpick/nitpick:latest
oc rollout restart deploy/nitpick
```
