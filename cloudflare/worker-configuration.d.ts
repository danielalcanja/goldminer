declare namespace Cloudflare {
  interface Env {
    API_TOKEN: string;
    OPENAI_API_KEY: string;
    R2_ACCESS_KEY_ID: string;
    R2_SECRET_ACCESS_KEY: string;
    R2_ENDPOINT_URL: string;
    R2_BUCKET: string;
    MEDIA_BUCKET: R2Bucket;
    GOLDMINER_CONTAINER: DurableObjectNamespace<import("./src/index").GoldMinerContainer>;
    GOLDMINER_WORKFLOW: Workflow<import("./src/index").JobRequest>;
  }
}
