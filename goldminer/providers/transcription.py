from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from goldminer.errors import GoldMinerError
from goldminer.media.ffmpeg import FFmpegTools, Runner, discover_tools, run_checked
from goldminer.transcript.normalize import normalize_text
from goldminer.transcript.models import Utterance


class TranscriptionProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def transcribe(self, audio_path: Path, media_duration_ms: int) -> Sequence[Utterance]: ...


@dataclass(frozen=True, slots=True)
class FakeTranscriptionProvider:
    utterances: Sequence[Utterance]
    provider_name: str = "fake"
    model_name: str = "fixture-v1"

    @property
    def name(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self.model_name

    def transcribe(self, audio_path: Path, media_duration_ms: int) -> Sequence[Utterance]:
        return tuple(self.utterances)


HttpTransport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


def _default_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except URLError as exc:
        raise GoldMinerError(f"Could not reach the OpenAI transcription API: {exc.reason}") from exc


def _multipart_body(fields: dict[str, str], file_path: Path) -> tuple[str, bytes]:
    boundary = f"goldminer-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(),
            b"Content-Type: audio/mpeg\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return boundary, b"".join(chunks)


@dataclass(frozen=True, slots=True)
class OpenAITranscriptionProvider:
    api_key: str
    model_name: str = "gpt-4o-transcribe-diarize"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 900.0
    max_retries: int = 3
    chunk_duration_seconds: int = 600
    transport: HttpTransport = _default_transport
    runner: Runner = subprocess.run
    tools: FFmpegTools | None = None
    progress: Callable[[str], None] | None = None

    @classmethod
    def from_environment(
        cls, *, progress: Callable[[str], None] | None = None
    ) -> "OpenAITranscriptionProvider":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            model_name=os.environ.get(
                "OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe-diarize"
            ).strip(),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            progress=progress,
        )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self.model_name

    def _prepare_upload(
        self,
        audio_path: Path,
        *,
        chunk_index: int,
        start_seconds: int,
        duration_seconds: int,
    ) -> Path:
        tools = self.tools or discover_tools()
        upload = audio_path.with_name(f".openai-transcription-{chunk_index:04d}.mp3")
        run_checked(
            [
                tools.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(start_seconds),
                "-i",
                str(audio_path),
                "-t",
                str(duration_seconds),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "32k",
                str(upload),
            ],
            runner=self.runner,
        )
        if not upload.is_file() or upload.stat().st_size == 0:
            raise GoldMinerError("FFmpeg did not create the OpenAI transcription upload artifact")
        if upload.stat().st_size > 25 * 1024 * 1024:
            upload.unlink(missing_ok=True)
            raise GoldMinerError(
                "Compressed audio exceeds the 25 MB transcription upload limit. "
                "Use a shorter source or provide an SRT/VTT transcript."
            )
        return upload

    def _request(self, upload: Path) -> dict[str, object]:
        boundary, body = _multipart_body(
            {
                "model": self.model_name,
                "response_format": "diarized_json",
                "chunking_strategy": "auto",
            },
            upload,
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "podcast-gold-miner/0.1.0",
        }
        last_status = 0
        last_payload = b""
        for attempt in range(self.max_retries + 1):
            try:
                status, payload = self.transport(
                    f"{self.base_url}/audio/transcriptions", headers, body, self.timeout_seconds
                )
            except (TimeoutError, URLError) as exc:
                status = 599
                payload = json.dumps({"error": {"message": f"network timeout: {exc}"}}).encode()
            if 200 <= status < 300:
                try:
                    decoded = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise GoldMinerError("OpenAI returned invalid JSON for the transcription") from exc
                if not isinstance(decoded, dict):
                    raise GoldMinerError("OpenAI transcription response must be a JSON object")
                return decoded
            last_status, last_payload = status, payload
            if status not in {408, 409, 429} and status < 500:
                break
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
        try:
            error_json = json.loads(last_payload)
            message = error_json.get("error", {}).get("message", last_payload.decode(errors="replace"))
        except (json.JSONDecodeError, AttributeError):
            message = last_payload.decode(errors="replace") or "unknown API error"
        raise GoldMinerError(f"OpenAI transcription failed with HTTP {last_status}: {message}")

    def _cache_key(self, audio_path: Path, start_seconds: int, duration_seconds: int) -> str:
        digest = hashlib.sha256()
        with audio_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(self.model_name.encode())
        digest.update(f":{start_seconds}:{duration_seconds}".encode())
        return digest.hexdigest()[:20]

    def _cache_path(
        self, audio_path: Path, chunk_index: int, start_seconds: int, duration_seconds: int
    ) -> Path:
        key = self._cache_key(audio_path, start_seconds, duration_seconds)
        return audio_path.parent / "transcription_cache" / f"chunk-{chunk_index:04d}-{key}.json"

    def _load_cached_response(self, path: Path) -> dict[str, object] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) and isinstance(value.get("segments"), list) else None

    def _save_cached_response(self, path: Path, response: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _segments_from_response(
        self,
        response: dict[str, object],
        *,
        offset_ms: int,
        media_duration_ms: int,
        first_id: int,
    ) -> list[Utterance]:
        raw_segments = response.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise GoldMinerError("OpenAI transcription returned no timestamped segments")
        utterances: list[Utterance] = []
        for local_index, segment in enumerate(raw_segments):
            index = first_id + local_index
            if not isinstance(segment, dict):
                raise GoldMinerError(f"OpenAI transcription segment {index} is malformed")
            try:
                start_ms = offset_ms + round(float(segment["start"]) * 1000)
                end_ms = offset_ms + round(float(segment["end"]) * 1000)
                text = str(segment["text"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise GoldMinerError(f"OpenAI transcription segment {index} is malformed") from exc
            if end_ms > media_duration_ms and end_ms - media_duration_ms <= 1000:
                end_ms = media_duration_ms
            if end_ms < start_ms:
                raise GoldMinerError(
                    f"OpenAI transcription segment {index} ends before it starts"
                )
            if end_ms == start_ms:
                # Diarization can timestamp a brief interjection (for example,
                # “Yes”) as a point event. Preserve its exact text and anchor,
                # while giving the canonical utterance a minimal valid span.
                if start_ms < media_duration_ms:
                    end_ms = min(media_duration_ms, start_ms + 250)
                elif start_ms > 0:
                    start_ms -= 1
            speaker_value = segment.get("speaker")
            speaker = str(speaker_value).strip() if speaker_value is not None else None
            utterances.append(
                Utterance(
                    id=f"u{index:06d}",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker=speaker or None,
                    original_text=text,
                    normalized_text=normalize_text(text),
                )
            )
        return utterances

    def transcribe(self, audio_path: Path, media_duration_ms: int) -> Sequence[Utterance]:
        if not self.api_key or self.api_key == "replace-with-new-key":
            raise GoldMinerError(
                "OPENAI_API_KEY is not configured. Add a new key to the project .env file "
                "or export it in your shell before running Gold Miner."
            )
        if self.chunk_duration_seconds < 1 or self.chunk_duration_seconds > 1400:
            raise GoldMinerError("OpenAI transcription chunk duration must be between 1 and 1400 seconds")

        utterances: list[Utterance] = []
        total_seconds = max(1, math.ceil(media_duration_ms / 1000))
        chunk_count = math.ceil(total_seconds / self.chunk_duration_seconds)
        for chunk_index in range(chunk_count):
            start_seconds = chunk_index * self.chunk_duration_seconds
            duration_seconds = min(self.chunk_duration_seconds, total_seconds - start_seconds)
            cache_path = self._cache_path(
                audio_path, chunk_index + 1, start_seconds, duration_seconds
            )
            response = self._load_cached_response(cache_path)
            if response is not None:
                if self.progress:
                    self.progress(f"Transcription chunk {chunk_index + 1}/{chunk_count}: reused")
                utterances.extend(
                    self._segments_from_response(
                        response,
                        offset_ms=start_seconds * 1000,
                        media_duration_ms=media_duration_ms,
                        first_id=len(utterances) + 1,
                    )
                )
                continue
            if self.progress:
                self.progress(f"Transcription chunk {chunk_index + 1}/{chunk_count}: uploading")
            upload = self._prepare_upload(
                audio_path,
                chunk_index=chunk_index + 1,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )
            try:
                response = self._request(upload)
                self._save_cached_response(cache_path, response)
                utterances.extend(
                    self._segments_from_response(
                        response,
                        offset_ms=start_seconds * 1000,
                        media_duration_ms=media_duration_ms,
                        first_id=len(utterances) + 1,
                    )
                )
            finally:
                upload.unlink(missing_ok=True)
            if self.progress:
                self.progress(f"Transcription chunk {chunk_index + 1}/{chunk_count}: completed")
        return tuple(utterances)
