"""Translate wire failures without allowing the framework to replay a stream."""

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError

from ._api import NariError, error_kind


def api_error(exc: Exception) -> APIError:
    # Safe pre-stream retries already happened in the wire client. Framework
    # retries would restart a whole sentence/stream, potentially repeating audio.
    if isinstance(exc, NariError) and exc.status:
        return APIStatusError(
            str(exc),
            status_code=exc.status,
            request_id=exc.request_id,
            body={"code": exc.code},
            retryable=False,
        )
    if isinstance(exc, TimeoutError) or isinstance(exc, NariError) and exc.code == "FINAL_TIMEOUT":
        return APITimeoutError(str(exc), retryable=False)
    if error_kind(exc) == "connectivity":
        return APIConnectionError(str(exc), retryable=False)
    body = {"code": exc.code, "request_id": exc.request_id} if isinstance(exc, NariError) else None
    return APIError(str(exc), body=body, retryable=False)
