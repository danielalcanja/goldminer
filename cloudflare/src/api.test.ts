import { describe, expect, it } from "vitest";

import {
  ApiError,
  createUpload,
  listClips,
  normalizeJobStatus,
  readJsonBody,
  validateUploadId,
  type ApiBindings,
} from "./api";

function signingBindings(writes: Map<string, string>): ApiBindings {
  return {
    R2_BUCKET: "goldminer-media",
    R2_ENDPOINT_URL: "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
    R2_ACCESS_KEY_ID: "A".repeat(32),
    R2_SECRET_ACCESS_KEY: "B".repeat(64),
    MEDIA_BUCKET: {
      async put(key: string, value: string) {
        writes.set(key, value);
        return {};
      },
    },
  } as unknown as ApiBindings;
}

describe("upload API", () => {
  it("creates a scoped presigned upload and stores its server-side record", async () => {
    const writes = new Map<string, string>();
    const request = new Request("https://api.example/v1/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: "My interview.mov",
        content_type: "video/quicktime",
        size_bytes: 12345,
      }),
    });

    const response = await createUpload(request, signingBindings(writes));
    const body = await response.json() as {
      upload_id: string;
      upload: { method: string; url: string; headers: Record<string, string> };
    };

    expect(response.status).toBe(201);
    expect(body.upload.method).toBe("PUT");
    expect(body.upload.headers["Content-Type"]).toBe("video/quicktime");
    expect(body.upload.url).toContain("X-Amz-Signature=");
    expect(body.upload.url).toContain(`/goldminer-media/uploads/api/${body.upload_id}/My_interview.mov`);
    const stored = writes.get(`uploads/api/${body.upload_id}/upload.json`);
    expect(stored).toBeDefined();
    expect(JSON.parse(stored ?? "{}")).toMatchObject({
      upload_id: body.upload_id,
      status: "pending",
      size_bytes: 12345,
    });
  });

  it("rejects an unsupported file before signing it", async () => {
    const request = new Request("https://api.example/v1/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: "notes.pdf",
        content_type: "application/pdf",
        size_bytes: 100,
      }),
    });

    await expect(createUpload(request, signingBindings(new Map()))).rejects.toMatchObject({
      code: "unsupported_video",
      status: 400,
    });
  });
});

describe("clip delivery", () => {
  it("returns short-lived direct URLs that a video element can play without an API token", async () => {
    const base = signingBindings(new Map());
    const bindings = {
      ...base,
      MEDIA_BUCKET: {
        async head() {
          return {};
        },
        async list() {
          return {
            objects: [{
              key: "jobs/job-1/artifacts/clips/01_lesson_92.mp4",
              size: 999,
              uploaded: new Date("2026-09-13T00:00:00Z"),
            }],
            truncated: false,
          };
        },
      },
    } as unknown as ApiBindings;

    const response = await listClips("job-1", bindings);
    const body = await response.json() as {
      clips: Array<{ playback_url: string; download_url: string; url_expires_at: string }>;
    };

    expect(body.clips).toHaveLength(1);
    expect(body.clips[0].playback_url).toContain("X-Amz-Signature=");
    expect(body.clips[0].download_url).toBe(body.clips[0].playback_url);
    expect(new Date(body.clips[0].url_expires_at).getTime()).toBeGreaterThan(Date.now());
  });
});

describe("request validation", () => {
  it("accepts UUID upload IDs and rejects arbitrary object keys", () => {
    expect(validateUploadId("123e4567-e89b-42d3-a456-426614174000"))
      .toBe("123e4567-e89b-42d3-a456-426614174000");
    expect(() => validateUploadId("uploads/private/video.mov")).toThrow(ApiError);
  });

  it("rejects oversized JSON before parsing", async () => {
    const request = new Request("https://api.example/v1/jobs", {
      method: "POST",
      headers: { "Content-Length": "65537", "Content-Type": "application/json" },
      body: "{}",
    });
    await expect(readJsonBody(request)).rejects.toMatchObject({
      code: "request_too_large",
      status: 413,
    });
  });

  it("rejects valid JSON that is not a request object", async () => {
    const request = new Request("https://api.example/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "null",
    });
    await expect(readJsonBody(request)).rejects.toMatchObject({
      code: "invalid_json",
      status: 400,
    });
  });
});

describe("frontend job status", () => {
  it("turns internal processing state into a useful customer message", () => {
    const status = normalizeJobStatus(
      "job-1",
      { status: "running" },
      { status: "running", phase: "processing", updated_at: "2026-09-13T00:00:00Z" },
      "https://api.example/v1/jobs/job-1",
    );
    expect(status).toMatchObject({
      status: "processing",
      progress: {
        phase: "processing",
        message: "Finding and editing the strongest moments",
      },
    });
  });

  it("does not expose raw workflow failures to the frontend", () => {
    const status = normalizeJobStatus(
      "job-1",
      { status: "errored", error: { message: "secret internal stack" } },
      null,
      "https://api.example/v1/jobs/job-1",
    );
    expect(status.status).toBe("failed");
    expect(JSON.stringify(status)).not.toContain("secret internal stack");
  });

  it("includes the clips endpoint after completion", () => {
    const status = normalizeJobStatus(
      "job-1",
      { status: "complete" },
      { status: "completed", phase: "completed" },
      "https://api.example/v1/jobs/job-1",
    );
    expect(status).toMatchObject({
      status: "completed",
      clips_url: "https://api.example/v1/jobs/job-1/clips",
    });
  });
});
