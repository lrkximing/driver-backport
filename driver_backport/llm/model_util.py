#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import selectors
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from driver_backport.core.common import int_or_zero as _integer


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_RATE_LIMIT_RETRY_DELAY_SEC = 60.0

# libcurl reports transport failures as stable numeric exit codes.  These
# failures occur before an HTTP response exists and are safe to retry.
RETRYABLE_CURL_EXIT_CODES = frozenset(
    {
        5,   # failed to resolve proxy
        6,   # failed to resolve host
        7,   # failed to connect
        16,  # HTTP/2 framing error
        18,  # partial transfer
        28,  # operation timeout
        35,  # TLS connection failure
        52,  # empty server response
        55,  # send failure
        56,  # receive failure
        92,  # HTTP/2 stream error
    }
)


class ModelAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        status_code: int | None = None,
        provider_code: str | None = None,
        provider_error: dict[str, Any] | None = None,
        retry_after_sec: float | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.provider_code = provider_code
        self.provider_error = dict(provider_error or {})
        self.retry_after_sec = (
            max(0.0, float(retry_after_sec))
            if retry_after_sec is not None
            else None
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "provider_code": self.provider_code,
            "provider_error": self.provider_error,
            "retry_after_sec": self.retry_after_sec,
        }


def model_api_error_metadata(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, ModelAPIError):
        return exc.metadata()
    return {
        "category": "unexpected_runtime_error",
        "retryable": False,
        "status_code": None,
        "provider_code": None,
        "provider_error": {},
        "retry_after_sec": None,
    }


def _parse_retry_after_seconds(
    value: object,
) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        deadline = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    current = datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (deadline - current).total_seconds())


def _retry_after_from_header_text(header_text: str) -> float | None:
    retry_after: float | None = None
    for line in str(header_text or "").splitlines():
        name, separator, value = line.partition(":")
        if not separator or name.strip().lower() != "retry-after":
            continue
        parsed = _parse_retry_after_seconds(value)
        if parsed is not None:
            retry_after = parsed
    return retry_after


def model_api_retry_delay_seconds(
    exc: BaseException,
    *,
    retry_index: int,
    rate_limit_fallback_sec: float = DEFAULT_RATE_LIMIT_RETRY_DELAY_SEC,
) -> float:
    exponent = min(3, max(0, int(retry_index)))
    exponential_delay = float(2**exponent)
    if not isinstance(exc, ModelAPIError):
        return exponential_delay
    if exc.retry_after_sec is not None:
        return exc.retry_after_sec
    if exc.status_code == 429:
        return max(exponential_delay, max(0.0, float(rate_limit_fallback_sec)))
    return exponential_delay


def _provider_error_record(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error")
    return dict(error) if isinstance(error, dict) else {}


def _provider_error_code(error: dict[str, Any]) -> str | None:
    value = error.get("code") or error.get("type")
    return str(value).strip() if value not in (None, "") else None


def _is_context_window_error(
    *,
    provider_code: str | None,
    provider_message: str | None,
) -> bool:
    normalized_code = str(provider_code or "").strip().lower()
    if normalized_code in {
        "context_length_exceeded",
        "context_window_exceeded",
        "input_too_long",
        "max_context_length_exceeded",
        "request_too_large",
    }:
        return True
    normalized_message = " ".join(
        str(provider_message or "").strip().lower().split()
    )
    return any(
        marker in normalized_message
        for marker in (
            "context length exceeded",
            "context window exceeded",
            "maximum context length",
            "maximum context window",
            "input is too long",
            "request is too large for this model",
            "too many input tokens",
        )
    )


def _is_retryable_provider_failure(
    *,
    provider_code: str | None,
    provider_message: str | None,
) -> bool:
    normalized_code = str(provider_code or "").strip().lower()
    if normalized_code in {
        "internal_server_error",
        "rate_limit_exceeded",
        "server_error",
        "server_overloaded",
        "service_unavailable",
        "upstream_error",
    }:
        return True
    normalized_message = " ".join(
        str(provider_message or "").strip().lower().split()
    )
    return any(
        marker in normalized_message
        for marker in (
            "abnormal closure",
            "stream closed before response.completed",
            "stream disconnected before completion",
            "unexpected eof",
            "websocket: close 1006",
        )
    )


def _api_error_category(
    *,
    status_code: int | None,
    provider_code: str | None,
    provider_message: str | None = None,
) -> tuple[str, bool]:
    normalized_code = str(provider_code or "").strip().lower()
    if _is_context_window_error(
        provider_code=normalized_code,
        provider_message=provider_message,
    ):
        return "context_window_exceeded", False
    if normalized_code in {"insufficient_quota", "insufficient_user_quota"}:
        return "insufficient_quota", False
    if normalized_code in {"invalid_api_key", "authentication_error"}:
        return "auth_error", False
    if normalized_code == "model_not_found":
        return "model_not_found", False
    normalized_message = " ".join(
        str(provider_message or "").strip().lower().split()
    )
    if any(
        marker in normalized_message
        for marker in (
            "out of memory",
            "memory allocation failed",
            "memory exhausted",
        )
    ):
        return "provider_resource_exhausted", False
    if _is_retryable_provider_failure(
        provider_code=normalized_code,
        provider_message=provider_message,
    ):
        return "retryable_runtime_error", True
    if status_code in {401, 403}:
        return "auth_error", False
    if status_code == 429 or status_code == 408:
        return "retryable_runtime_error", True
    if status_code is not None and 500 <= status_code <= 599:
        return "retryable_runtime_error", True
    return "non_retryable_runtime_error", False


def _structured_output_unsupported(
    exc: ModelAPIError,
    *,
    response_schema_present: bool,
) -> bool:
    if not response_schema_present or exc.status_code not in {400, 404, 422}:
        return False
    detail = (
        json.dumps(exc.provider_error, ensure_ascii=False)
        + " "
        + str(exc)
    ).lower()
    return any(
        marker in detail
        for marker in (
            "response_format",
            "response format",
            "json_schema",
            "json schema",
            "structured output",
        )
    )


def _curl_transport_error(
    *,
    operation: str,
    returncode: int,
    detail: str,
) -> ModelAPIError:
    retryable = returncode in RETRYABLE_CURL_EXIT_CODES
    return ModelAPIError(
        f"{operation} failed with exit code {returncode}: {detail}",
        category=(
            "retryable_runtime_error"
            if retryable
            else "non_retryable_runtime_error"
        ),
        retryable=retryable,
    )


def _raise_provider_error(
    *,
    status_code: int | None,
    body_text: str,
    retry_after_sec: float | None = None,
) -> None:
    try:
        payload = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    error = _provider_error_record(payload)
    provider_code = _provider_error_code(error)
    message = str(error.get("message") or body_text or "model API request failed").strip()
    category, retryable = _api_error_category(
        status_code=status_code,
        provider_code=provider_code,
        provider_message=message,
    )
    raise ModelAPIError(
        message,
        category=category,
        retryable=retryable,
        status_code=status_code,
        provider_code=provider_code,
        provider_error=error,
        retry_after_sec=retry_after_sec,
    )


def _candidate_config_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("DRIVER_BACKPORT_MODEL_CONFIG")
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.append(Path(__file__).resolve().parents[1] / "local.model.json")
    paths.append(Path.home() / ".config" / "driver_backport" / "model.json")
    return paths


def _load_file_config() -> dict[str, Any]:
    for path in _candidate_config_paths():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"failed to read model config {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"model config must be a JSON object: {path}")
        return payload
    return {}


def _first_config_value(file_config: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    for name in names:
        value = file_config.get(name)
        if value not in (None, ""):
            return value
    return default


def _boolean_config_value(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"invalid boolean model configuration {name}={value!r}")


def get_model_config() -> dict[str, Any]:
    file_config = _load_file_config()
    temperature = _first_config_value(
        file_config,
        "OPENAI_TEMPERATURE",
        "temperature",
    )
    return {
        "api_key": _first_config_value(file_config, "OPENAI_API_KEY", "api_key"),
        "base_url": _first_config_value(
            file_config,
            "OPENAI_BASE_URL",
            "base_url",
            default=DEFAULT_BASE_URL,
        ),
        "model": _first_config_value(
            file_config,
            "OPENAI_MODEL",
            "model",
            default=DEFAULT_MODEL,
        ),
        "stream": _boolean_config_value(
            _first_config_value(
                file_config,
                "OPENAI_STREAM",
                "stream",
                default=False,
            ),
            name="stream",
        ),
        "system_prompt": _first_config_value(
            file_config,
            "OPENAI_SYSTEM_PROMPT",
            "system_prompt",
            default="",
        ),
        "timeout_sec": float(
            _first_config_value(
                file_config,
                "OPENAI_TIMEOUT_SEC",
                "timeout_sec",
                default=180,
            )
        ),
        "max_retries": int(
            _first_config_value(
                file_config,
                "OPENAI_MAX_RETRIES",
                "max_retries",
                default=2,
            )
        ),
        "temperature": (
            float(temperature)
            if temperature not in (None, "")
            else None
        ),
        "max_completion_tokens": int(
            _first_config_value(
                file_config,
                "OPENAI_MAX_COMPLETION_TOKENS",
                "max_completion_tokens",
                default=0,
            )
            or 0
        ),
        "max_context_tokens": int(
            _first_config_value(
                file_config,
                "OPENAI_MAX_CONTEXT_TOKENS",
                "max_context_tokens",
                default=0,
            )
            or 0
        ),
        "token_count_url": str(
            _first_config_value(
                file_config,
                "OPENAI_TOKEN_COUNT_URL",
                "token_count_url",
                default="",
            )
            or ""
        ).strip(),
        "rate_limit_retry_delay_sec": float(
            _first_config_value(
                file_config,
                "OPENAI_RATE_LIMIT_RETRY_DELAY_SEC",
                "rate_limit_retry_delay_sec",
                default=DEFAULT_RATE_LIMIT_RETRY_DELAY_SEC,
            )
        ),
        "pricing_currency": str(
            _first_config_value(
                file_config,
                "OPENAI_PRICING_CURRENCY",
                "pricing_currency",
                default="",
            )
            or ""
        ).strip(),
        "input_price_per_million_tokens": float(
            _first_config_value(
                file_config,
                "OPENAI_INPUT_PRICE_PER_MILLION_TOKENS",
                "input_price_per_million_tokens",
                default=0,
            )
            or 0
        ),
        "output_price_per_million_tokens": float(
            _first_config_value(
                file_config,
                "OPENAI_OUTPUT_PRICE_PER_MILLION_TOKENS",
                "output_price_per_million_tokens",
                default=0,
            )
            or 0
        ),
        "cached_input_price_per_million_tokens": float(
            _first_config_value(
                file_config,
                "OPENAI_CACHED_INPUT_PRICE_PER_MILLION_TOKENS",
                "cached_input_price_per_million_tokens",
                default=0,
            )
            or 0
        ),
    }


def has_model_config() -> bool:
    return bool(get_model_config().get("api_key"))


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _usage_metadata(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("usage")
    raw = raw if isinstance(raw, dict) else {}
    input_details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or raw.get("completion_tokens_details") or {}
    input_tokens = _integer(raw.get("input_tokens", raw.get("prompt_tokens")))
    output_tokens = _integer(raw.get("output_tokens", raw.get("completion_tokens")))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _integer(raw.get("total_tokens")) or input_tokens + output_tokens,
        "cached_input_tokens": _integer(
            input_details.get("cached_tokens") if isinstance(input_details, dict) else 0
        ),
        "reasoning_tokens": _integer(
            output_details.get("reasoning_tokens") if isinstance(output_details, dict) else 0
        ),
    }


def _provider_reported_cost(payload: dict[str, Any]) -> float | None:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    for container in (payload, usage):
        for key in ("cost", "total_cost", "charge", "fee", "price"):
            value = _number(container.get(key))
            if value is not None:
                return value
    return None


def _cost_metadata(config: dict[str, Any], usage: dict[str, int], payload: dict[str, Any]) -> dict[str, Any]:
    provider_reported = _provider_reported_cost(payload)
    input_price = float(config.get("input_price_per_million_tokens") or 0)
    output_price = float(config.get("output_price_per_million_tokens") or 0)
    cached_price = float(config.get("cached_input_price_per_million_tokens") or 0)
    cached_tokens = min(usage["cached_input_tokens"], usage["input_tokens"])
    uncached_tokens = max(0, usage["input_tokens"] - cached_tokens)
    estimated = None
    if input_price or output_price or cached_price:
        estimated = (
            uncached_tokens * input_price
            + cached_tokens * (cached_price or input_price)
            + usage["output_tokens"] * output_price
        ) / 1_000_000
    return {
        "currency": str(config.get("pricing_currency") or "") or None,
        "provider_reported": provider_reported,
        "estimated": estimated,
        "estimation_source": (
            "configured_per_million_token_prices" if estimated is not None else None
        ),
        "input_price_per_million_tokens": input_price or None,
        "output_price_per_million_tokens": output_price or None,
        "cached_input_price_per_million_tokens": cached_price or None,
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_streaming_process(
    command: list[str],
    *,
    idle_timeout_sec: float,
    operation: str,
) -> subprocess.CompletedProcess[str]:
    """Drain a streaming subprocess and time out only on inactivity."""

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        _terminate_process(process)
        raise RuntimeError("streaming subprocess pipes are unavailable")

    output = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    last_activity = time.monotonic()
    timeout = max(0.0, float(idle_timeout_sec))
    try:
        while selector.get_map():
            remaining = timeout - (time.monotonic() - last_activity)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            events = selector.select(remaining)
            if not events:
                raise subprocess.TimeoutExpired(command, timeout)
            for key, _mask in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    output[str(key.data)].extend(chunk)
                    last_activity = time.monotonic()
                    continue
                selector.unregister(key.fileobj)

        remaining = timeout - (time.monotonic() - last_activity)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        _terminate_process(process)
        raise ModelAPIError(
            f"{operation} timed out after no streaming data",
            category="retryable_runtime_error",
            retryable=True,
        ) from exc
    except BaseException:
        _terminate_process(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()

    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=bytes(output["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(output["stderr"]).decode("utf-8", errors="replace"),
    )


def _openai_compatible_json_request(
    *,
    config: dict[str, Any],
    url: str,
    body: dict[str, Any],
    operation: str,
    stream: bool = False,
) -> dict[str, Any]:
    curl = os.environ.get("DRIVER_BACKPORT_CURL", "curl")
    with tempfile.TemporaryDirectory(prefix="driver-backport-model-") as tmp:
        tmp_path = Path(tmp)
        config_path = tmp_path / "curl.conf"
        body_path = tmp_path / "body.json"
        header_path = tmp_path / "response-headers.txt"
        body_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        config_path.write_text(
            "\n".join(
                [
                    f'url = "{url}"',
                    f'header = "Authorization: Bearer {config["api_key"]}"',
                    'header = "Content-Type: application/json"',
                    f"data = @{body_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            os.chmod(config_path, 0o600)
            os.chmod(body_path, 0o600)
        except OSError:
            pass
        command = [
            curl,
            "-sS",
        ]
        if stream:
            command.append("--no-buffer")
        command.extend(
            [
                "--dump-header",
                str(header_path),
                "-w",
                "\nHTTP_STATUS:%{http_code}\n",
                "-K",
                str(config_path),
            ]
        )
        if stream:
            result = _run_streaming_process(
                command,
                idle_timeout_sec=float(config["timeout_sec"]),
                operation=operation,
            )
        else:
            try:
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=float(config["timeout_sec"]),
                )
            except subprocess.TimeoutExpired as exc:
                raise ModelAPIError(
                    f"{operation} timed out",
                    category="retryable_runtime_error",
                    retryable=True,
                ) from exc
        retry_after_sec = _retry_after_from_header_text(
            header_path.read_text(encoding="utf-8", errors="replace")
            if header_path.is_file()
            else ""
        )
    output = result.stdout
    body_text, marker, status_text = output.rpartition("\nHTTP_STATUS:")
    status_text = status_text.strip() if marker else ""
    if result.returncode != 0:
        detail = (result.stderr or body_text or "").strip()
        raise _curl_transport_error(
            operation=operation,
            returncode=result.returncode,
            detail=detail,
        )
    if status_text and not status_text.startswith("2"):
        _raise_provider_error(
            status_code=_integer(status_text) or None,
            body_text=body_text,
            retry_after_sec=retry_after_sec,
        )
    raw = body_text
    if stream:
        return _openai_chat_completion_stream_payload(
            raw,
            operation=operation,
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelAPIError(
            f"{operation} returned non-JSON response: {raw[:500]}",
            category="non_retryable_runtime_error",
            retryable=False,
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelAPIError(
            f"{operation} returned unexpected response type: {type(parsed).__name__}",
            category="non_retryable_runtime_error",
            retryable=False,
        )
    if isinstance(parsed.get("error"), dict):
        _raise_provider_error(status_code=None, body_text=raw)
    return parsed


def _openai_chat_completion_stream_payload(
    raw: str,
    *,
    operation: str,
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    data_lines: list[str] = []
    done = False

    def consume_event() -> None:
        nonlocal done
        if not data_lines:
            return
        data = "\n".join(data_lines).strip()
        data_lines.clear()
        if data == "[DONE]":
            done = True
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ModelAPIError(
                f"{operation} returned malformed SSE data: {data[:500]}",
                category="retryable_runtime_error",
                retryable=True,
            ) from exc
        if not isinstance(payload, dict):
            raise ModelAPIError(
                f"{operation} returned unexpected SSE payload type: "
                f"{type(payload).__name__}",
                category="retryable_runtime_error",
                retryable=True,
            )
        if isinstance(payload.get("error"), dict):
            _raise_provider_error(status_code=None, body_text=data)
        chunks.append(payload)

    for raw_line in str(raw or "").splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            consume_event()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data" and separator:
            data_lines.append(value.lstrip())
            continue
        if field in {"event", "id", "retry"} and separator:
            continue
        raise ModelAPIError(
            f"{operation} returned malformed SSE line: {line[:500]}",
            category="retryable_runtime_error",
            retryable=True,
        )
    consume_event()

    response: dict[str, Any] = {}
    content_parts: list[str] = []
    role = "assistant"
    finish_reason: object = None
    for chunk in chunks:
        for key in ("id", "model", "object", "created", "system_fingerprint"):
            if key not in response and chunk.get(key) is not None:
                response[key] = chunk[key]
        if isinstance(chunk.get("usage"), dict):
            response["usage"] = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict) or _integer(choice.get("index")) != 0:
                continue
            message = choice.get("delta")
            if not isinstance(message, dict):
                message = choice.get("message")
            if not isinstance(message, dict):
                message = {}
            if message.get("role"):
                role = str(message["role"])
            content = message.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ModelAPIError(
                        f"{operation} returned non-text SSE content",
                        category="non_retryable_runtime_error",
                        retryable=False,
                    )
                content_parts.append(content)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    if not chunks or not done or finish_reason is None:
        raise ModelAPIError(
            f"{operation} returned an incomplete SSE response",
            category="retryable_runtime_error",
            retryable=True,
        )
    response["choices"] = [
        {
            "index": 0,
            "message": {
                "role": role,
                "content": "".join(content_parts),
            },
            "finish_reason": finish_reason,
        }
    ]
    return response


def _openai_compatible_chat_completion(
    *,
    config: dict[str, Any],
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    response_schema: dict[str, Any] | None = None,
    response_schema_name: str = "structured_response",
) -> dict[str, Any]:
    base_url = str(config["base_url"]).rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if max_completion_tokens > 0:
        body["max_tokens"] = max_completion_tokens
    if config.get("temperature") is not None:
        body["temperature"] = float(config["temperature"])
    if response_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(response_schema_name),
                "strict": True,
                "schema": response_schema,
            },
        }
    stream = bool(config.get("stream"))
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return _openai_compatible_json_request(
        config=config,
        url=f"{base_url}/chat/completions",
        body=body,
        operation="model API chat completion",
        stream=stream,
    )


def _model_request_budget(
    *,
    config: dict[str, Any],
    model: str,
    messages: list[dict[str, str]],
    requested_max_completion_tokens: int,
) -> dict[str, Any]:
    requested = max(0, int(requested_max_completion_tokens))
    configured_cap = max(0, int(config.get("max_completion_tokens") or 0))
    effective = (
        min(requested, configured_cap)
        if requested and configured_cap
        else configured_cap or requested
    )
    prompt_tokens: int | None = None
    token_count_url = str(config.get("token_count_url") or "").strip()
    if token_count_url:
        token_count = _openai_compatible_json_request(
            config=config,
            url=token_count_url,
            body={"model": model, "messages": messages},
            operation="model API prompt token count",
        )
        prompt_tokens = _integer(token_count.get("input_tokens"))
        if prompt_tokens <= 0:
            raise ModelAPIError(
                "model token-count response did not contain a positive input_tokens value",
                category="non_retryable_runtime_error",
                retryable=False,
            )

    configured_context_tokens = max(
        0,
        int(config.get("max_context_tokens") or 0),
    )
    context_tokens = configured_context_tokens or None
    if context_tokens is not None:
        if prompt_tokens is None:
            raise ModelAPIError(
                "max_context_tokens requires an exact token_count_url",
                category="non_retryable_runtime_error",
                retryable=False,
            )
        if effective <= 0:
            raise ModelAPIError(
                "max_context_tokens requires a positive completion-token limit",
                category="non_retryable_runtime_error",
                retryable=False,
            )
        if prompt_tokens + effective > context_tokens:
            raise ModelAPIError(
                (
                    "model request exceeds the configured context window: "
                    f"prompt_tokens={prompt_tokens} "
                    f"max_completion_tokens={effective} "
                    f"max_context_tokens={context_tokens}"
                ),
                category="context_window_exceeded",
                retryable=False,
                provider_error={
                    "prompt_tokens": prompt_tokens,
                    "max_completion_tokens": effective,
                    "max_context_tokens": context_tokens,
                },
            )
    return {
        "prompt_tokens": prompt_tokens,
        "requested_max_completion_tokens": requested,
        "configured_max_completion_tokens": configured_cap or None,
        "effective_max_completion_tokens": effective,
        "max_context_tokens": context_tokens,
    }


def _chat_completion_with_retries(
    *,
    config: dict[str, Any],
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    max_retries: int,
    response_schema: dict[str, Any] | None = None,
    response_schema_name: str = "structured_response",
) -> tuple[dict[str, Any], int]:
    retry_count = 0
    while True:
        try:
            return (
                _openai_compatible_chat_completion(
                    config=config,
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    response_schema=response_schema,
                    response_schema_name=response_schema_name,
                ),
                retry_count,
            )
        except ModelAPIError as exc:
            if not exc.retryable or retry_count >= max_retries:
                raise
            configured_rate_limit_delay = config.get(
                "rate_limit_retry_delay_sec"
            )
            time.sleep(
                model_api_retry_delay_seconds(
                    exc,
                    retry_index=retry_count,
                    rate_limit_fallback_sec=float(
                        DEFAULT_RATE_LIMIT_RETRY_DELAY_SEC
                        if configured_rate_limit_delay is None
                        else configured_rate_limit_delay
                    ),
                )
            )
            retry_count += 1


def gpt_api_with_metadata(
    prompt: str,
    *,
    model: str | None = None,
    max_completion_tokens: int = 8192,
    max_retries: int | None = None,
    response_schema: dict[str, Any] | None = None,
    response_schema_name: str = "structured_response",
) -> dict[str, Any]:
    config = get_model_config()
    if not config["api_key"]:
        raise ModelAPIError(
            "model API key is not set; use OPENAI_API_KEY, "
            "DRIVER_BACKPORT_MODEL_CONFIG, driver_backport/local.model.json, "
            "or ~/.config/driver_backport/model.json",
            category="auth_error",
            retryable=False,
        )
    selected_model = model or config["model"]
    if not selected_model:
        raise ModelAPIError(
            "model name is not set; use --model, OPENAI_MODEL, "
            "DRIVER_BACKPORT_MODEL_CONFIG, driver_backport/local.model.json, "
            "or ~/.config/driver_backport/model.json",
            category="model_not_found",
            retryable=False,
        )

    system_prompt = str(config.get("system_prompt") or "").strip()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    request_budget = _model_request_budget(
        config=config,
        model=selected_model,
        messages=messages,
        requested_max_completion_tokens=max_completion_tokens,
    )
    effective_max_completion_tokens = int(
        request_budget["effective_max_completion_tokens"]
    )
    configured_retries = config["max_retries"] if max_retries is None else max(0, int(max_retries))
    structured_output_mode = (
        "strict_json_schema"
        if response_schema is not None
        else "text"
    )
    structured_output_error: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        payload, retry_count = _chat_completion_with_retries(
            config=config,
            model=selected_model,
            messages=messages,
            max_completion_tokens=effective_max_completion_tokens,
            max_retries=configured_retries,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
        )
    except ModelAPIError as exc:
        schema_unsupported = _structured_output_unsupported(
            exc,
            response_schema_present=response_schema is not None,
        )
        if not schema_unsupported:
            raise
        structured_output_error = exc.metadata()
        payload, retry_count = _chat_completion_with_retries(
            config=config,
            model=selected_model,
            messages=messages,
            max_completion_tokens=effective_max_completion_tokens,
            max_retries=configured_retries,
        )
        structured_output_mode = (
            "text_fallback_provider_rejected_json_schema"
        )
    elapsed_sec = time.perf_counter() - started
    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = _usage_metadata(payload)
    return {
        "text": str(message.get("content") or ""),
        "model": payload.get("model") or selected_model,
        "api_mode": "chat",
        "stream": bool(config.get("stream")),
        "response_id": payload.get("id"),
        "retry_count": retry_count,
        "request_budget": request_budget,
        "usage": usage,
        "cost": _cost_metadata(config, usage, payload),
        "elapsed_sec": elapsed_sec,
        "structured_output_mode": structured_output_mode,
        "structured_output_error": structured_output_error,
    }
