import { Container, getContainer } from "@cloudflare/containers";
import {
  env,
  WorkflowEntrypoint,
  type WorkflowEvent,
  type WorkflowStep,
} from "cloudflare:workers";

const JOB_ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,99}$/;
const OBJECT_KEY_PATTERN = /^(?!\/)(?!.*(?:^|\/)\.\.?(?:\/|$))(?!.*\\)[^\0]+$/;
const CONTAINER_SLOTS = 3;

export type JobRequest = {
  schema_version: "1.0";
  job_id: string;
  source_key: string;
  result_prefix: string;
  transcript_key?: string;
  top_k: number;
  max_duration_seconds: number;
  handle_ms: number;
};

type Bindings = Cloudflare.Env;

type ContainerSuccess = {
  schema_version: string;
  job_id: string;
  status: string;
  status_key: string;
  artifacts: Array<{ name: string; key: string; size_bytes: number }>;
};

type ContainerFailure = { error: string; message?: string };

type CreateJobBody = {
  source_key?: unknown;
  transcript_key?: unknown;
  top_k?: unknown;
  max_duration_seconds?: unknown;
  handle_ms?: unknown;
};

function json(value: unknown, status = 200): Response {
  return Response.json(value, {
    status,
    headers: { "Cache-Control": "no-store" },
  });
}

async function containerResult(
  response: Response,
): Promise<ContainerSuccess | ContainerFailure> {
  const body = await response.text();
  let result: unknown;
  try {
    result = JSON.parse(body);
  } catch {
    const detail = body.trim().replace(/\s+/g, " ").slice(0, 500) || "empty response";
    throw new Error(
      `Container request returned HTTP ${response.status} with a non-JSON response: ${detail}`,
    );
  }
  if (!result || typeof result !== "object") {
    throw new Error(`Container returned HTTP ${response.status} with an invalid JSON response`);
  }
  return result as ContainerSuccess | ContainerFailure;
}

function authorized(request: Request, secret: string): boolean {
  return request.headers.get("Authorization") === `Bearer ${secret}`;
}

function objectKey(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim() || !OBJECT_KEY_PATTERN.test(value.trim())) {
    throw new Error(`${field} must be a safe relative R2 object key`);
  }
  return value.trim();
}

function integer(
  value: unknown,
  fallback: number,
  field: string,
  minimum: number,
  maximum: number,
): number {
  const result = value === undefined ? fallback : value;
  if (!Number.isInteger(result) || (result as number) < minimum || (result as number) > maximum) {
    throw new Error(`${field} must be an integer between ${minimum} and ${maximum}`);
  }
  return result as number;
}

function parseJob(body: CreateJobBody, jobId: string): JobRequest {
  const transcriptKey =
    body.transcript_key === undefined
      ? undefined
      : objectKey(body.transcript_key, "transcript_key");
  return {
    schema_version: "1.0",
    job_id: jobId,
    source_key: objectKey(body.source_key, "source_key"),
    result_prefix: `jobs/${jobId}`,
    transcript_key: transcriptKey,
    top_k: integer(body.top_k, 8, "top_k", 1, 25),
    max_duration_seconds: integer(
      body.max_duration_seconds,
      60,
      "max_duration_seconds",
      15,
      60,
    ),
    handle_ms: integer(body.handle_ms, 500, "handle_ms", 0, 5000),
  };
}

function containerSlot(jobId: string): string {
  let hash = 2166136261;
  for (const character of jobId) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return `slot-${(hash >>> 0) % CONTAINER_SLOTS}`;
}

async function r2Json(bucket: R2Bucket, key: string): Promise<unknown | null> {
  const object = await bucket.get(key);
  return object ? object.json() : null;
}

export class GoldMinerContainer extends Container {
  defaultPort = 8080;
  requiredPorts = [8080];
  sleepAfter = "4h";
  envVars = {
    OPENAI_API_KEY: env.OPENAI_API_KEY,
    R2_ACCESS_KEY_ID: env.R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY: env.R2_SECRET_ACCESS_KEY,
    R2_ENDPOINT_URL: env.R2_ENDPOINT_URL,
    R2_BUCKET: env.R2_BUCKET,
    GOLDMINER_WORK_ROOT: "/work/jobs",
  };
}

export class GoldMinerWorkflow extends WorkflowEntrypoint<Bindings, JobRequest> {
  async run(event: WorkflowEvent<JobRequest>, step: WorkflowStep) {
    return step.do(
      "process video with Gold Miner",
      {
        retries: { limit: 1, delay: "30 seconds", backoff: "exponential" },
        timeout: "6 hours",
      },
      async () => {
        const container = getContainer<GoldMinerContainer>(
          this.env.GOLDMINER_CONTAINER,
          containerSlot(event.instanceId),
        );
        await container.startAndWaitForPorts({
          ports: 8080,
          cancellationOptions: {
            instanceGetTimeoutMS: 120_000,
            portReadyTimeoutMS: 120_000,
            waitInterval: 500,
          },
        });
        const response = await container.fetch(
          new Request("http://container/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(event.payload),
          }),
        );
        const result = await containerResult(response);
        if (!response.ok) {
          const message =
            "message" in result && result.message
              ? result.message
              : "error" in result && result.error
                ? result.error
                : "Container job failed";
          throw new Error(`Container returned HTTP ${response.status}: ${message}`);
        }
        if (!("artifacts" in result)) {
          throw new Error("Container returned an invalid success response");
        }
        return result;
      },
    );
  }
}

async function createJob(request: Request, bindings: Bindings): Promise<Response> {
  let body: CreateJobBody;
  try {
    body = await request.json<CreateJobBody>();
  } catch {
    return json({ error: "Request body must be valid JSON" }, 400);
  }

  const jobId = crypto.randomUUID();
  let job: JobRequest;
  try {
    job = parseJob(body, jobId);
  } catch (error) {
    return json({ error: error instanceof Error ? error.message : "Invalid job" }, 400);
  }

  if (!(await bindings.MEDIA_BUCKET.head(job.source_key))) {
    return json({ error: `Source object not found: ${job.source_key}` }, 404);
  }
  if (job.transcript_key && !(await bindings.MEDIA_BUCKET.head(job.transcript_key))) {
    return json({ error: `Transcript object not found: ${job.transcript_key}` }, 404);
  }

  const queued = {
    schema_version: "1.0",
    job_id: jobId,
    status: "queued",
    source_key: job.source_key,
    updated_at: new Date().toISOString(),
  };
  await Promise.all([
    bindings.MEDIA_BUCKET.put(`${job.result_prefix}/request.json`, JSON.stringify(job), {
      httpMetadata: { contentType: "application/json" },
    }),
    bindings.MEDIA_BUCKET.put(`${job.result_prefix}/status.json`, JSON.stringify(queued), {
      httpMetadata: { contentType: "application/json" },
    }),
  ]);

  try {
    const instance = await bindings.GOLDMINER_WORKFLOW.create({ id: jobId, params: job });
    return json(
      {
        job_id: instance.id,
        status: "queued",
        status_url: new URL(`/jobs/${instance.id}`, request.url).toString(),
      },
      202,
    );
  } catch (error) {
    const failed = {
      ...queued,
      status: "failed",
      updated_at: new Date().toISOString(),
      error: error instanceof Error ? error.message : "Could not create workflow",
    };
    await bindings.MEDIA_BUCKET.put(
      `${job.result_prefix}/status.json`,
      JSON.stringify(failed),
      { httpMetadata: { contentType: "application/json" } },
    );
    return json({ error: failed.error }, 500);
  }
}

async function getJob(jobId: string, bindings: Bindings): Promise<Response> {
  if (!JOB_ID_PATTERN.test(jobId)) {
    return json({ error: "Invalid job ID" }, 400);
  }
  try {
    const instance = await bindings.GOLDMINER_WORKFLOW.get(jobId);
    const [workflow, worker] = await Promise.all([
      instance.status(),
      r2Json(bindings.MEDIA_BUCKET, `jobs/${jobId}/status.json`),
    ]);
    return json({ job_id: jobId, workflow, worker });
  } catch {
    return json({ error: "Job not found" }, 404);
  }
}

async function listArtifacts(jobId: string, bindings: Bindings): Promise<Response> {
  if (!JOB_ID_PATTERN.test(jobId)) {
    return json({ error: "Invalid job ID" }, 400);
  }
  const prefix = `jobs/${jobId}/artifacts/`;
  const listing = await bindings.MEDIA_BUCKET.list({ prefix });
  return json({
    job_id: jobId,
    artifacts: listing.objects.map((object) => ({
      name: object.key.slice(prefix.length),
      key: object.key,
      size_bytes: object.size,
      uploaded_at: object.uploaded.toISOString(),
    })),
    truncated: listing.truncated,
  });
}

async function getArtifact(jobId: string, name: string, bindings: Bindings): Promise<Response> {
  if (!JOB_ID_PATTERN.test(jobId) || !name || !OBJECT_KEY_PATTERN.test(name)) {
    return json({ error: "Invalid artifact path" }, 400);
  }
  const object = await bindings.MEDIA_BUCKET.get(`jobs/${jobId}/artifacts/${name}`);
  if (!object) {
    return json({ error: "Artifact not found" }, 404);
  }
  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("ETag", object.httpEtag);
  headers.set("Cache-Control", "private, max-age=3600");
  return new Response(object.body, { headers });
}

export default {
  async fetch(request: Request, bindings: Bindings): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/healthz") {
      return json({ status: "ok" });
    }
    if (!authorized(request, bindings.API_TOKEN)) {
      return json({ error: "Unauthorized" }, 401);
    }
    if (request.method === "POST" && url.pathname === "/jobs") {
      return createJob(request, bindings);
    }

    const match = url.pathname.match(/^\/jobs\/([^/]+)(?:\/artifacts(?:\/(.+))?)?$/);
    if (!match || request.method !== "GET") {
      return json({ error: "Not found" }, 404);
    }
    const [, jobId, artifactName] = match;
    if (url.pathname.endsWith("/artifacts")) {
      return listArtifacts(jobId, bindings);
    }
    if (artifactName) {
      return getArtifact(jobId, artifactName, bindings);
    }
    return getJob(jobId, bindings);
  },
};
