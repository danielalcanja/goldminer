FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOLDMINER_WORK_ROOT=/work/jobs

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates ffmpeg tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system goldminer \
    && useradd --system --gid goldminer --create-home goldminer

WORKDIR /app
COPY pyproject.toml README.md ./
COPY goldminer ./goldminer
RUN python -m pip install --no-cache-dir '.[worker]' \
    && mkdir -p /work/jobs \
    && chown -R goldminer:goldminer /work

USER goldminer
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["goldminer-worker", "serve", "--host", "0.0.0.0", "--port", "8080"]
