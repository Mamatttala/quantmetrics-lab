"""Typed application errors and their HTTP mapping.

Errors carry a machine-readable `code` so the frontend can branch on failure kind
(e.g. show a date-range picker for SPAN_TOO_WIDE) instead of string-matching prose.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class QuantmetricsError(Exception):
    """Base class for all deliberate application failures."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ValidationFailure(QuantmetricsError):
    """Request is well-formed but the parameters are not usable."""

    status_code = 422
    code = "validation_failure"


class SpanTooWide(QuantmetricsError):
    """A single request exceeded the provider's permitted window.

    Raised instead of silently truncating: a quietly short frame looks like a
    successful fetch and corrupts every downstream statistic.
    """

    status_code = 422
    code = "span_too_wide"


class UnsupportedRequest(QuantmetricsError):
    """The provider does not serve this combination of parameters."""

    status_code = 422
    code = "unsupported_request"


class DatasetNotFound(QuantmetricsError):
    status_code = 404
    code = "dataset_not_found"


class JobNotFound(QuantmetricsError):
    """No job with that id.

    A separate type from ``DatasetNotFound`` so the frontend can say "that run does not
    exist" rather than "that dataset does not exist" — they are reached from different
    screens, and a wrong noun sends the reader looking in the wrong place.
    """

    status_code = 404
    code = "job_not_found"


class ImmutableDataset(QuantmetricsError):
    """Attempted in-place write to a raw or uploaded dataset."""

    status_code = 409
    code = "immutable_dataset"


class UpstreamError(QuantmetricsError):
    """The remote provider failed or rate limited us."""

    status_code = 502
    code = "upstream_error"


#: The prefixes pydantic puts in front of a message that came out of a validator. Stripped
#: so the envelope's message is the sentence the validator wrote rather than a sentence
#: about it: ``"Value error, Select at least one model to fit."`` is a message that reads
#: like a leak from the validation library in a panel the user is looking at.
_VALIDATOR_PREFIXES = ("Value error, ", "Assertion failed, ")

#: The request parts pydantic puts at the head of a location. Dropped so a refusal about a
#: field reads as ``("test_window: ...")`` rather than ``("body → test_window: ...")``.
_LOCATION_ROOTS = frozenset({"body", "query", "path", "header", "cookie"})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(QuantmetricsError)
    async def _handle(_: Request, exc: QuantmetricsError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Every 422 leaves in the same envelope, whoever rejected the request.

        ``QuantmetricsError`` covers a refusal a *service* reached. This covers the ones
        pydantic reaches first — a field out of range, a missing key, a ``model_validator``
        inside a request model — which FastAPI answers with ``{"detail": [...]}`` unless
        something says otherwise. They are not a rare path: ``app/models/regime.py``,
        ``scaling.py``, ``splits.py`` and ``volatility.py`` all raise from a validator, so
        this is how most malformed requests are refused. Without this handler those
        responses carry no ``error.code``, and ``src/api/client.ts`` — which builds its
        ``ApiError`` from that code and renders ``error.message`` — degrades to a generic
        failure for exactly the refusals written to be specific.

        The location is folded into the message because the envelope has one message field
        and naming the field is the actionable half of a refusal. ``details.errors`` keeps
        the structured list for a caller that wants to mark several fields at once. It is
        rebuilt field by field rather than passed through from ``exc.errors()``: pydantic
        puts the original exception object under ``ctx``, and this payload is rendered with
        ``allow_nan=False``, which raises on anything the encoder cannot represent.
        """
        errors = list(exc.errors())
        first = errors[0] if errors else {}
        message = str(first.get("msg", "The request could not be validated."))
        for prefix in _VALIDATOR_PREFIXES:
            if message.startswith(prefix):
                message = message[len(prefix) :]
                break
        location = " → ".join(
            str(part) for part in first.get("loc", ()) if str(part) not in _LOCATION_ROOTS
        )
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=ValidationFailure.status_code,
            content={
                "error": {
                    "code": ValidationFailure.code,
                    "message": message,
                    "details": {
                        "errors": [
                            {
                                "loc": [str(part) for part in error.get("loc", ())],
                                "msg": str(error.get("msg", "")),
                                "type": str(error.get("type", "")),
                            }
                            for error in errors
                        ],
                        "count": len(errors),
                    },
                }
            },
        )
