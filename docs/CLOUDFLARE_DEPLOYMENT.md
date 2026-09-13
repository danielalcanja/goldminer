# Gold Miner on Cloudflare

The hosted layer runs the unchanged Gold Miner CLI inside a Cloudflare Container. API v1 gives a frontend a safe upload-to-results workflow; it is still a backend service rather than the customer-facing page or email-delivery layer.

## Architecture

```text
frontend server
        |
        | creates an upload session and starts/polls a job
        v
Cloudflare Worker ---- signs upload, reads status/clips ----> R2 bucket
        ^                                                   ^
        |                                                   |
        +------ browser uploads video directly by PUT ------+
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

The app-facing v1 API accepts no filesystem paths or arbitrary R2 keys. It accepts an API-created `upload_id`, verifies the uploaded object's size and type, then starts one idempotent job for that upload. The source object is read-only, every job writes beneath `jobs/<job-id>/`, and hosted clip duration is capped at 60 seconds. The original `/jobs` endpoints remain available for trusted operational clients.

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

Set the allowed frontend origins in `CORS_ORIGINS` in `wrangler.jsonc`, and put the same origins in `cloudflare/r2-cors.json`. Apply the R2 rule after changing it:

```bash
npx wrangler r2 bucket cors set goldminer-media \
  --file cloudflare/r2-cors.json
```

The API Bearer token belongs only in your frontend's server-side environment. Never include it in browser JavaScript. The browser receives only a short-lived R2 URL scoped to one object.

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

## Frontend API workflow

The complete contract is in [`docs/OPENAPI.yaml`](OPENAPI.yaml). A frontend uses this sequence:

1. Its server calls `POST /v1/uploads` with the selected file's name, browser MIME type, and exact byte size.
2. The browser uploads the file directly to the returned `upload.url` with `PUT` and the returned headers.
3. Its server calls `POST /v1/uploads/<upload-id>/complete`.
4. Its server calls `POST /v1/jobs` with the `upload_id`.
5. It polls `GET /v1/jobs/<job-id>` and displays the returned progress message.
6. When complete, it calls `GET /v1/jobs/<job-id>/clips` and uses each one-hour playback or download URL. Calling the list endpoint again refreshes expired URLs.

Create an upload session from the frontend server:

```bash
curl -X POST "$GOLDMINER_API_URL/v1/uploads" \
  -H "Authorization: Bearer $GOLDMINER_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"filename":"interview.mov","content_type":"video/quicktime","size_bytes":12345678}'
```

The response includes a one-hour upload URL. The browser must use the exact returned method and headers:

```js
await fetch(upload.url, {
  method: upload.method,
  headers: upload.headers,
  body: videoFile,
});
```

After that upload finishes, verify it and start the job:

```bash
curl -X POST "$GOLDMINER_API_URL/v1/uploads/<upload-id>/complete" \
  -H "Authorization: Bearer $GOLDMINER_API_TOKEN"

curl -X POST "$GOLDMINER_API_URL/v1/jobs" \
  -H "Authorization: Bearer $GOLDMINER_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"upload_id":"<upload-id>"}'
```

## Legacy operational API

The trusted legacy routes accept an R2 source key directly. They remain available for operations and backwards compatibility.

## Run the first hosted job

To test the trusted legacy API, upload a video into R2 with Wrangler:

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
- API v1 supports a single direct PUT up to 5 GiB. Resumable multipart uploads are still needed for unreliable connections and videos larger than 5 GiB.
- User accounts, quotas, billing, email notification, retention/deletion, and abuse controls belong in the web-app milestone.
- The API token is intentionally simple for private testing. Do not expose this Worker publicly as a multi-user product until per-user authentication and authorization are added.
