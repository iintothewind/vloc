from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("vloc")

STATUS = {
    "invalid_request": 422,
    "image_too_large": 413,
    "unsupported_image": 415,
    "image_fetch_failed": 400,
    "not_found": 404,
    "not_ready": 503,
    "busy": 503,
    "timeout": 504,
    "internal": 500,
}


class ApiError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.status = STATUS[code]
        self.message = message
        self.details = details or {}


def _body(code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        headers = {"Retry-After": "5"} if exc.code == "busy" else None
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = {"errors": jsonable_encoder(exc.errors())}
        return JSONResponse(_body("invalid_request", "request validation failed", details), status_code=422)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(_body("internal", "internal error", {}), status_code=500)
