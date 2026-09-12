# Gold Miner on Cloudflare

This first hosted layer runs the unchanged Gold Miner CLI inside a Cloudflare Container. It is intentionally a backend job API, not yet the customer-facing upload page or email-delivery layer.

## Architecture

```text
website or API client
        |
        | POST /jobs with an existing R2 source key
        v
Cloudflare Worker ---- reads status/artifacts ----> R2 bucket
        |
        | starts one durable job
        v
Cloudflare Workflow
        |
        | assigns one of 3 processing slots and waits for POST /run
        v
Cloudflare Container
  1. downloads source from R2
  2. invokes python -m goldminer ... --resume
  3. uploads clips, report, manifest, and diagnostics to R2
```

The Workflow is important: a Queue consumer has a 15-minute wall-clock limit, while a Workflow step can wait for a long-running network request without a fixed wall-time limit. Jobs are deterministically assigned across three Container slots. Each slot processes one job at a time, and each job gets an isolated local directory.

## What is included

- `Dockerfile`: Python 3.12, FFmpeg/FFprobe, the Gold Miner package, and the R2 worker dependency.
- `goldminer-worker serve`: a small HTTP process used by the Container.
- `goldminer-worker run JOB.json`: a one-job command for local or operational testing.
- `cloudflare/src/index.ts`: authenticated job creation, job status, artifact listing/download, Workflow orchestration, and Container routing.
- `wrangler.jsonc`: R2, Workflow, Durable Object, and Container bindings.

The app-facing API accepts no filesystem paths. It accepts only validated R2 object keys. The source object is read-only, every job writes beneath `jobs/<job-id>/`, and hosted clip duration is capped at 60 seconds.

## Prerequisites

- A Cloudflare Workers Paid account. Cloudflare Containers require the paid Workers plan.
- Docker Desktop running locally for the first image build.
- Node.js 22 or newer (required by the current Wrangler release).
- An R2 API token scoped to read and write the `goldminer-media` bucket.
- An OpenAI API key.

Cloudflare Containers currently require `linux/amd64`. Wrangler handles that target when it builds this Dockerfile.

## Configure Cloudflare

Install the deployment dependencies:

```bash
npm install
```

Authenticate Wrangler and create the R2 bucket configured in `wrangler.jsonc`:

```bash
npx wrangler login
npx wrangler r2 bucket create goldminer-media
```

Create an R2 API token in the Cloudflare dashboard with Object Read & Write access restricted to that bucket. Save its Access Key ID and Secret Access Key. The endpoint has this form:

```text
https://<CLOUDFLARE_ACCOUNT_ID>.r2.cloudflarestorage.com
```

Add the five deployment secrets. Wrangler prompts for each value and does not put it in the repository:

```bash
npx wrangler secret put API_TOKEN
npx wrangler secret put OPENAI_API_KEY
npx wrangler secret put R2_ACCESS_KEY_ID
npx wrangler secret put R2_SECRET_ACCESS_KEY
npx wrangler secret put R2_ENDPOINT_URL
```

`API_TOKEN` is a new random value used to protect the temporary backend API. It is not a Cloudflare account token. The future website authentication layer should replace this single shared token before public launch.

## Verify and deploy

Check the TypeScript configuration, then deploy the Worker and image:

```bash
npm run check
npm run deploy
```

The first deploy takes longer because Wrangler builds and uploads the container image. The API can become reachable a few minutes before the new Container capacity is fully provisioned.

Check the public health endpoint:

```bash
curl https://goldminer-jobs.<your-workers-subdomain>.workers.dev/healthz
```

## Run the first hosted job

For this backend milestone, upload a test video into R2 with Wrangler. The web app will later replace this with browser-based multipart upload.

```bash
npx wrangler r2 object put \
  goldminer-media/uploads/demo/video.mov \
  --file=/absolute/path/to/video.mov \
  --remote
```

Set the deployed URL and the same API token you saved as the `API_TOKEN` Worker secret:

```bash
export GOLDMINER_API_URL="https://goldminer-jobs.<your-workers-subdomain>.workers.dev"
read -s GOLDMINER_API_TOKEN
export GOLDMINER_API_TOKEN
```

Start a job. The response is immediate and includes a `job_id` and `status_url`:

```bash
curl -X POST "$GOLDMINER_API_URL/jobs" \
  -H "Authorization: Bearer $GOLDMINER_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"source_key":"uploads/demo/video.mov"}'
```

Poll the returned status URL:

```bash
curl "$GOLDMINER_API_URL/jobs/<job-id>" \
  -H "Authorization: Bearer $GOLDMINER_API_TOKEN"
```

When its status is `complete`, list the generated artifacts:

```bash
curl "$GOLDMINER_API_URL/jobs/<job-id>/artifacts" \
  -H "Authorization: Bearer $GOLDMINER_API_TOKEN"
```

Download a listed clip:

```bash
curl "$GOLDMINER_API_URL/jobs/<job-id>/artifacts/clips/01_lesson_92.mp4" \
  -H "Authorization: Bearer $GOLDMINER_API_TOKEN" \
  --output 01_lesson_92.mp4
```

## Local container check

The image can be built before deployment:

```bash
docker build --platform linux/amd64 -t goldminer-worker .
docker run --rm -p 8080:8080 --env-file .env.worker goldminer-worker
curl http://localhost:8080/healthz
```

Use an ignored `.env.worker` file containing the OpenAI and R2 variables. Never put those values in the Dockerfile, `.env.example`, a job JSON document, or Git.

## Current operational limits

- `standard-3` provides 2 vCPU, 8 GiB RAM, and 16 GB ephemeral disk. The source video plus temporary audio and outputs must fit. Large uploads should be rejected or preprocessed before job creation, or the instance type should be raised.
- Container disk is ephemeral. A failed attempt keeps its job directory so the immediate Workflow retry can use Gold Miner's local `--resume` checkpoints. Cloudflare does not guarantee a Container will run for any fixed period, however, so a host interruption can still restart paid provider work. Persisting per-stage caches to R2 is the next reliability hardening step.
- The first API assumes the source already exists in R2. Browser multipart upload, user accounts, quotas, billing, email notification, retention/deletion, and abuse controls belong in the web-app milestone.
- The API token is intentionally simple for private testing. Do not expose this Worker publicly as a multi-user product until per-user authentication and authorization are added.
