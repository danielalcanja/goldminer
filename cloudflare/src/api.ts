import { AwsClient } from "aws4fetch";

const API_SCHEMA_VERSION = "1.0" as const;
const MAX_JSON_BYTES = 64 * 1024;
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024;
const UPLOAD_URL_TTL_SECONDS = 60 * 60;
const UPLOAD_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SAFE_CLIP_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.mp4$/i;
const SUPPORTED_VIDEO_TYPES = new Set([
  "video/mp4",
  "video/quicktime",
  "video/x-m4v",
  "video/webm",
]);

export type ApiBindings = Cloudflare.Env;

export type UploadRecord = {
  schema_version: typeof API_SCHEMA_VERSION;
  upload_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  source_key: string;
  status: "pending" | "completed";
  created_at: string;
  expires_at: string;
  completed_at?: string;
  job_id?: string;
};

export type V1CreateJobBody = {
  upload_id?: unknown;
  top_k?: unknown;
  max_duration_seconds?: unknown;
  handle_ms?: unknown;
};

type V1CreateUploadBody = {
  filename?: unknown;
  content_type?: unknown;
  size_bytes?: unknown;
};

type WorkerStatus = {
  status?: unknown;
  phase?: unknown;
  updated_at?: unknown;
  started_at?: unknown;
  completed_at?: unknown;
  failed_at?: unknown;
  error?: unknown;
};

type WorkflowStatus = {
  status?: unknown;
  error?: unknown;
};

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function apiJson(
  value: unknown,
  status = 200,
  extraHeaders?: HeadersInit,
): Response {
  const headers = new Headers(extraHeaders);
  headers.set("Cache-Control", "no-store");
  return Response.json(value, { status, headers });
}

export function apiErrorResponse(error: unknown, requestId: string): Response {
  if (error instanceof ApiError) {
    return apiJson(
      {
        error: {
          code: error.code,
          message: error.message,
          request_id: requestId,
          ...(error.details ? { details: error.details } : {}),
        },
      },
      error.status,
    );
  }
  console.error("Unhandled API error", error);
  return apiJson(
    {
      error: {
        code: "internal_error",
        message: "The service could not complete this request.",
        request_id: requestId,
      },
    },
    500,
  );
}

export async function readJsonBody<T>(request: Request): Promise<T> {
  const contentLength = request.headers.get("Content-Length");
  if (contentLength !== null) {
    const parsedLength = Number(contentLength);
    if (!Number.isFinite(parsedLength) || parsedLength < 0) {
      throw new ApiError(400, "invalid_content_length", "Content-Length is invalid.");
    }
    if (parsedLength > MAX_JSON_BYTES) {
      throw new ApiError(413, "request_too_large", "The JSON request body is too large.");
    }
  }
  const contentType = request.headers.get("Content-Type")?.split(";", 1)[0].trim();
  if (contentType && contentType !== "application/json") {
    throw new ApiError(415, "unsupported_media_type", "Content-Type must be application/json.");
  }
  let body: string;
  try {
    body = await request.text();
  } catch {
    throw new ApiError(400, "invalid_json", "Request body must be valid JSON.");
  }
  if (new TextEncoder().encode(body).byteLength > MAX_JSON_BYTES) {
    throw new ApiError(413, "request_too_large", "The JSON request body is too large.");
  }
  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch {
    throw new ApiError(400, "invalid_json", "Request body must be valid JSON.");
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new ApiError(400, "invalid_json", "Request body must be a JSON object.");
  }
  return value as T;
}

function requiredString(value: unknown, field: string, maximum: number): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new ApiError(400, "invalid_request", `${field} must be a non-empty string.`);
  }
  const result = value.trim();
  if (result.length > maximum) {
    throw new ApiError(400, "invalid_request", `${field} is too long.`);
  }
  return result;
}

function safeFilename(value: unknown): string {
  const filename = requiredString(value, "filename", 200).normalize("NFKC");
  if (filename.includes("/") || filename.includes("\\") || filename.includes("\0")) {
    throw new ApiError(400, "invalid_filename", "filename must not contain a path.");
  }
  const extension = filename.toLowerCase().match(/\.(mp4|mov|m4v|webm)$/);
  if (!extension) {
    throw new ApiError(
      400,
      "unsupported_video",
      "Use an MP4, MOV, M4V, or WebM video file.",
    );
  }
  const sanitized = filename
    .replace(/[^\p{L}\p{N}._-]+/gu, "_")
    .replace(/^\.+/, "")
    .slice(0, 200);
  if (!sanitized) {
    throw new ApiError(400, "invalid_filename", "filename is invalid.");
  }
  return sanitized;
}

function uploadSize(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new ApiError(400, "invalid_size", "size_bytes must be a positive integer.");
  }
  if ((value as number) > MAX_UPLOAD_BYTES) {
    throw new ApiError(
      413,
      "video_too_large",
      "This upload API currently supports videos up to 5 GiB.",
      { max_size_bytes: MAX_UPLOAD_BYTES },
    );
  }
  return value as number;
}

function contentType(value: unknown): string {
  const result = requiredString(value, "content_type", 100).toLowerCase();
  if (!SUPPORTED_VIDEO_TYPES.has(result)) {
    throw new ApiError(
      400,
      "unsupported_video",
      "Use an MP4, MOV, M4V, or WebM video file.",
      { supported_content_types: [...SUPPORTED_VIDEO_TYPES] },
    );
  }
  return result;
}

function encodeObjectPath(key: string): string {
  return key.split("/").map(encodeURIComponent).join("/");
}

function r2Endpoint(bindings: ApiBindings): URL {
  let endpoint: URL;
  try {
    endpoint = new URL(bindings.R2_ENDPOINT_URL);
  } catch {
    throw new ApiError(500, "service_misconfigured", "Video storage is not configured.");
  }
  if (endpoint.protocol !== "https:" || !endpoint.hostname.endsWith(".r2.cloudflarestorage.com")) {
    throw new ApiError(500, "service_misconfigured", "Video storage is not configured.");
  }
  return endpoint;
}

function r2Signer(bindings: ApiBindings): AwsClient {
  return new AwsClient({
    accessKeyId: bindings.R2_ACCESS_KEY_ID,
    secretAccessKey: bindings.R2_SECRET_ACCESS_KEY,
    service: "s3",
    region: "auto",
  });
}

async function presignedR2Url(
  key: string,
  method: "GET" | "PUT",
  bindings: ApiBindings,
  headers?: HeadersInit,
): Promise<string> {
  const endpoint = r2Endpoint(bindings);
  const url = new URL(
    `${endpoint.origin}/${encodeURIComponent(bindings.R2_BUCKET)}/${encodeObjectPath(key)}`,
  );
  url.searchParams.set("X-Amz-Expires", String(UPLOAD_URL_TTL_SECONDS));
  const signed = await r2Signer(bindings).sign(
    new Request(url, { method, headers }),
    { aws: { signQuery: true } },
  );
  return signed.url;
}

function uploadRecordKey(uploadId: string): string {
  return `uploads/api/${uploadId}/upload.json`;
}

export function validateUploadId(value: unknown): string {
  if (typeof value !== "string" || !UPLOAD_ID_PATTERN.test(value)) {
    throw new ApiError(400, "invalid_upload_id", "upload_id is invalid.");
  }
  return value.toLowerCase();
}

export async function getUpload(
  uploadId: string,
  bindings: ApiBindings,
): Promise<UploadRecord | null> {
  const object = await bindings.MEDIA_BUCKET.get(uploadRecordKey(uploadId));
  if (!object) return null;
  try {
    return await object.json<UploadRecord>();
  } catch {
    throw new ApiError(500, "upload_state_invalid", "The upload state is invalid.");
  }
}

async function putUpload(record: UploadRecord, bindings: ApiBindings): Promise<void> {
  await bindings.MEDIA_BUCKET.put(uploadRecordKey(record.upload_id), JSON.stringify(record), {
    httpMetadata: { contentType: "application/json" },
  });
}

export async function createUpload(
  request: Request,
  bindings: ApiBindings,
): Promise<Response> {
  const body = await readJsonBody<V1CreateUploadBody>(request);
  const filename = safeFilename(body.filename);
  const requestedContentType = contentType(body.content_type);
  const sizeBytes = uploadSize(body.size_bytes);
  const uploadId = crypto.randomUUID();
  const now = new Date();
  const expires = new Date(now.getTime() + UPLOAD_URL_TTL_SECONDS * 1000);
  const sourceKey = `uploads/api/${uploadId}/${filename}`;
  const record: UploadRecord = {
    schema_version: API_SCHEMA_VERSION,
    upload_id: uploadId,
    filename,
    content_type: requestedContentType,
    size_bytes: sizeBytes,
    source_key: sourceKey,
    status: "pending",
    created_at: now.toISOString(),
    expires_at: expires.toISOString(),
  };

  await putUpload(record, bindings);
  const signedUrl = await presignedR2Url(
    sourceKey,
    "PUT",
    bindings,
    { "Content-Type": requestedContentType },
  );

  return apiJson(
    {
      upload_id: uploadId,
      status: "pending",
      upload: {
        method: "PUT",
        url: signedUrl,
        headers: { "Content-Type": requestedContentType },
        expires_at: record.expires_at,
      },
      complete_url: new URL(`/v1/uploads/${uploadId}/complete`, request.url).toString(),
      max_size_bytes: MAX_UPLOAD_BYTES,
    },
    201,
  );
}

export async function completeUpload(
  uploadIdValue: string,
  bindings: ApiBindings,
): Promise<Response> {
  const uploadId = validateUploadId(uploadIdValue);
  const record = await getUpload(uploadId, bindings);
  if (!record) {
    throw new ApiError(404, "upload_not_found", "Upload not found.");
  }
  if (record.status === "completed") {
    return apiJson({
      upload_id: uploadId,
      status: "completed",
      size_bytes: record.size_bytes,
    });
  }
  const object = await bindings.MEDIA_BUCKET.head(record.source_key);
  if (!object) {
    throw new ApiError(409, "upload_incomplete", "The video has not finished uploading.");
  }
  if (object.size !== record.size_bytes) {
    throw new ApiError(
      409,
      "upload_size_mismatch",
      "The uploaded video size does not match the requested upload.",
      { expected_size_bytes: record.size_bytes, actual_size_bytes: object.size },
    );
  }
  const storedType = object.httpMetadata?.contentType?.toLowerCase();
  if (storedType && storedType !== record.content_type) {
    throw new ApiError(
      409,
      "upload_type_mismatch",
      "The uploaded video type does not match the requested upload.",
    );
  }
  const completed: UploadRecord = {
    ...record,
    status: "completed",
    completed_at: new Date().toISOString(),
  };
  await putUpload(completed, bindings);
  return apiJson({
    upload_id: uploadId,
    status: "completed",
    size_bytes: object.size,
  });
}

export async function assignUploadJob(
  uploadId: string,
  jobId: string,
  bindings: ApiBindings,
): Promise<void> {
  const record = await getUpload(uploadId, bindings);
  if (!record) {
    throw new ApiError(404, "upload_not_found", "Upload not found.");
  }
  await putUpload({ ...record, job_id: jobId }, bindings);
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export function normalizeJobStatus(
  jobId: string,
  workflowValue: unknown,
  workerValue: unknown,
  requestUrl: string,
): Record<string, unknown> {
  const workflow = (recordValue(workflowValue) ?? {}) as WorkflowStatus;
  const worker = (recordValue(workerValue) ?? {}) as WorkerStatus;
  const workflowStatus = stringValue(workflow.status);
  const workerStatus = stringValue(worker.status);
  const workerPhase = stringValue(worker.phase);

  let status: "queued" | "processing" | "completed" | "failed" = "queued";
  let phase = "queued";
  let message = "Waiting to start";
  if (workerStatus === "completed" || workflowStatus === "complete") {
    status = "completed";
    phase = "completed";
    message = "Your clips are ready";
  } else if (workerStatus === "failed" || workflowStatus === "errored" || workflowStatus === "terminated") {
    status = "failed";
    phase = "failed";
    message = "GoldMiner could not process this video";
  } else if (workerStatus === "running" || ["running", "waiting"].includes(workflowStatus ?? "")) {
    status = "processing";
    phase = workerPhase ?? "starting";
    message =
      phase === "downloading"
        ? "Preparing your video"
        : phase === "uploading"
          ? "Preparing your clips"
          : phase === "processing"
            ? "Finding and editing the strongest moments"
            : "Starting GoldMiner";
  }

  const result: Record<string, unknown> = {
    job_id: jobId,
    status,
    progress: { phase, message },
    created_at: stringValue(worker.started_at) ?? stringValue(worker.updated_at),
    updated_at: stringValue(worker.updated_at),
  };
  if (status === "completed") {
    result.completed_at = stringValue(worker.completed_at);
    result.clips_url = new URL(`/v1/jobs/${jobId}/clips`, requestUrl).toString();
  } else if (status === "failed") {
    result.failed_at = stringValue(worker.failed_at) ?? stringValue(worker.updated_at);
    result.error = {
      code: "processing_failed",
      message: "GoldMiner could not process this video. Please try a different video or contact support.",
      retryable: false,
    };
  }
  return result;
}

function safeClipName(value: string): string {
  let decoded: string;
  try {
    decoded = decodeURIComponent(value);
  } catch {
    throw new ApiError(400, "invalid_clip_id", "clip_id is invalid.");
  }
  if (!SAFE_CLIP_PATTERN.test(decoded)) {
    throw new ApiError(400, "invalid_clip_id", "clip_id is invalid.");
  }
  return decoded;
}

export async function listClips(
  jobId: string,
  bindings: ApiBindings,
): Promise<Response> {
  if (!(await bindings.MEDIA_BUCKET.head(`jobs/${jobId}/request.json`))) {
    throw new ApiError(404, "job_not_found", "Job not found.");
  }
  const prefix = `jobs/${jobId}/artifacts/clips/`;
  const listing = await bindings.MEDIA_BUCKET.list({ prefix, limit: 100 });
  const expiresAt = new Date(Date.now() + UPLOAD_URL_TTL_SECONDS * 1000).toISOString();
  const clips = await Promise.all(
    listing.objects
      .filter((object) => object.key.toLowerCase().endsWith(".mp4"))
      .map(async (object) => {
        const name = object.key.slice(prefix.length);
        const signedUrl = await presignedR2Url(object.key, "GET", bindings);
        return {
          clip_id: name,
          filename: name,
          size_bytes: object.size,
          created_at: object.uploaded.toISOString(),
          playback_url: signedUrl,
          download_url: signedUrl,
          url_expires_at: expiresAt,
        };
      }),
  );
  return apiJson({ job_id: jobId, clips, count: clips.length });
}

function rangeHeaders(object: R2ObjectBody, headers: Headers): number {
  headers.set("Accept-Ranges", "bytes");
  const range = object.range;
  if (!range) {
    headers.set("Content-Length", String(object.size));
    return 200;
  }
  let offset: number;
  let length: number;
  if ("suffix" in range) {
    length = Math.min(range.suffix, object.size);
    offset = object.size - length;
  } else {
    offset = range.offset ?? 0;
    length = range.length ?? object.size - offset;
  }
  headers.set("Content-Length", String(length));
  headers.set("Content-Range", `bytes ${offset}-${offset + length - 1}/${object.size}`);
  return 206;
}

export async function getClip(
  jobId: string,
  clipId: string,
  request: Request,
  bindings: ApiBindings,
): Promise<Response> {
  const name = safeClipName(clipId);
  let object: R2ObjectBody | null;
  try {
    object = await bindings.MEDIA_BUCKET.get(`jobs/${jobId}/artifacts/clips/${name}`, {
      range: request.headers,
    });
  } catch {
    throw new ApiError(416, "invalid_range", "The requested video range is invalid.");
  }
  if (!object) {
    throw new ApiError(404, "clip_not_found", "Clip not found.");
  }
  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("Content-Type", object.httpMetadata?.contentType ?? "video/mp4");
  headers.set("ETag", object.httpEtag);
  headers.set("Cache-Control", "private, max-age=3600");
  const disposition = new URL(request.url).searchParams.get("download") === "1"
    ? "attachment"
    : "inline";
  headers.set("Content-Disposition", `${disposition}; filename="${name.replace(/[\"\\]/g, "_")}"`);
  const status = rangeHeaders(object, headers);
  return new Response(object.body, { status, headers });
}

function allowedOrigins(bindings: ApiBindings): Set<string> {
  return new Set(
    (bindings.CORS_ORIGINS ?? "")
      .split(",")
      .map((origin) => origin.trim())
      .filter(Boolean),
  );
}

export function addCors(request: Request, response: Response, bindings: ApiBindings): Response {
  const origin = request.headers.get("Origin");
  if (!origin || !allowedOrigins(bindings).has(origin)) return response;
  const result = new Response(response.body, response);
  result.headers.set("Access-Control-Allow-Origin", origin);
  result.headers.set("Access-Control-Allow-Credentials", "true");
  result.headers.set("Access-Control-Expose-Headers", "Content-Length, Content-Range, ETag");
  result.headers.append("Vary", "Origin");
  return result;
}

export function corsPreflight(request: Request, bindings: ApiBindings): Response {
  const origin = request.headers.get("Origin");
  if (!origin || !allowedOrigins(bindings).has(origin)) {
    return apiJson(
      { error: { code: "origin_not_allowed", message: "This origin is not allowed." } },
      403,
    );
  }
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": origin,
      "Access-Control-Allow-Credentials": "true",
      "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Max-Age": "86400",
      Vary: "Origin",
    },
  });
}
