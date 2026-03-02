"""OpenDAP plugin for xpublish.

Provides DAP2-conformant access to xarray Datasets via REST endpoints.
All handlers are async with constraint-first subsetting and memory management.
"""

from __future__ import annotations

import logging
from urllib.parse import unquote

import xarray as xr
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from xpublish import Dependencies, Plugin, hookimpl

from xpublish_opendap.dap.constraint import Constraint, parse_dap2_constraint
from xpublish_opendap.dap.dap2.das import generate_das
from xpublish_opendap.dap.dap2.dds import generate_dds
from xpublish_opendap.dap.dap2.dods import generate_dods
from xpublish_opendap.dap.dap2.headers import CONTENT_DESCRIPTIONS, CONTENT_TYPES, DAP2_HEADERS
from xpublish_opendap.dap.dap2.responses import generate_error, generate_help, generate_version
from xpublish_opendap.dap.xarray_adapter import apply_plan, plan_subsetting
from xpublish_opendap.errors import DapError, RequestTooLargeError
from xpublish_opendap.io import load_dataset_async

logger: logging.Logger = logging.getLogger('xpublish_opendap')


class OpenDapPlugin(Plugin):
    """OpenDAP plugin for xpublish."""

    name: str = 'opendap'

    dataset_router_prefix: str = '/opendap'
    dataset_router_tags: list[str] = ['opendap']

    # Configuration
    max_request_memory_bytes: int = 512 * 1024 * 1024  # 512 MB
    num_concurrent_data_loads: int = 4
    async_load_timeout: float = 30.0

    @hookimpl
    def dataset_router(self, deps: Dependencies) -> APIRouter:
        """Create an OpenDAP router for xpublish."""
        router = APIRouter(
            prefix=self.dataset_router_prefix,
            tags=self.dataset_router_tags,
        )

        config = self

        def _extract_constraint(request: Request) -> str:
            """Extract the constraint expression from the request URL query."""
            query = request.url.query or ''
            return unquote(query)

        def _parse_constraint(raw: str) -> Constraint:
            """Parse a raw constraint string into a Constraint object."""
            if not raw:
                return Constraint()
            return parse_dap2_constraint(raw)

        def _dap2_headers(response_type: str) -> dict[str, str]:
            """Build DAP2 response headers."""
            headers = dict(DAP2_HEADERS)
            if response_type in CONTENT_DESCRIPTIONS:
                headers['Content-Description'] = CONTENT_DESCRIPTIONS[response_type]
            return headers

        def _error_response(error: DapError, status_code: int = 400) -> Response:
            """Build a DAP2 error response."""
            body = ''.join(generate_error(error.code, error.message))
            return Response(
                content=body,
                media_type=CONTENT_TYPES['error'],
                status_code=status_code,
                headers=_dap2_headers('error'),
            )

        @router.get('.dds')
        async def dds_response(
            request: Request,
            dataset_id: str = 'default',
            ds: xr.Dataset = Depends(deps.dataset),
        ) -> Response:
            """DAP2 Dataset Descriptor Structure response."""
            try:
                raw_constraint = _extract_constraint(request)
                constraint = _parse_constraint(raw_constraint)
                plan = plan_subsetting(ds, constraint)
                subsetted = apply_plan(ds, plan)

                body = ''.join(generate_dds(subsetted, dataset_id))
                return Response(
                    content=body,
                    media_type=CONTENT_TYPES['dds'],
                    headers=_dap2_headers('dds'),
                )
            except DapError as e:
                return _error_response(e)
            except Exception:
                logger.exception('Unexpected error in DDS handler')
                return _error_response(
                    DapError(code=500, message='Internal server error'),
                    status_code=500,
                )

        @router.get('.das')
        async def das_response(
            request: Request,
            ds: xr.Dataset = Depends(deps.dataset),
        ) -> Response:
            """DAP2 Dataset Attribute Structure response."""
            try:
                raw_constraint = _extract_constraint(request)
                constraint = _parse_constraint(raw_constraint)
                plan = plan_subsetting(ds, constraint)
                subsetted = apply_plan(ds, plan)

                body = ''.join(generate_das(subsetted))
                return Response(
                    content=body,
                    media_type=CONTENT_TYPES['das'],
                    headers=_dap2_headers('das'),
                )
            except DapError as e:
                return _error_response(e)
            except Exception:
                logger.exception('Unexpected error in DAS handler')
                return _error_response(
                    DapError(code=500, message='Internal server error'),
                    status_code=500,
                )

        @router.get('.dods', response_model=None)
        async def dods_response(
            request: Request,
            dataset_id: str = 'default',
            ds: xr.Dataset = Depends(deps.dataset),
        ) -> StreamingResponse | Response:
            """DAP2 DataDDS (binary data) response."""
            try:
                raw_constraint = _extract_constraint(request)
                constraint = _parse_constraint(raw_constraint)
                plan = plan_subsetting(ds, constraint)

                # Memory threshold check
                if plan.estimated_bytes > config.max_request_memory_bytes:
                    raise RequestTooLargeError(
                        plan.estimated_bytes,
                        config.max_request_memory_bytes,
                    )

                subsetted = apply_plan(ds, plan)

                # Async load the data
                loaded = await load_dataset_async(
                    subsetted,
                    timeout=config.async_load_timeout,
                )

                return StreamingResponse(
                    generate_dods(loaded, dataset_id),
                    media_type=CONTENT_TYPES['dods'],
                    headers=_dap2_headers('dods'),
                )
            except DapError as e:
                status = 413 if isinstance(e, RequestTooLargeError) else 400
                return _error_response(e, status_code=status)
            except Exception:
                logger.exception('Unexpected error in DODS handler')
                return _error_response(
                    DapError(code=500, message='Internal server error'),
                    status_code=500,
                )

        @router.get('.ver')
        async def version_response() -> Response:
            """DAP2 Version response."""
            body = ''.join(generate_version())
            return Response(
                content=body,
                media_type=CONTENT_TYPES['version'],
                headers=_dap2_headers('dds'),
            )

        @router.get('.help')
        async def help_response() -> HTMLResponse:
            """DAP2 Help response."""
            return HTMLResponse(
                content=generate_help(),
                headers=_dap2_headers('dds'),
            )

        @router.get('')
        async def root_help_response() -> HTMLResponse:
            """Bare path redirects to help."""
            return HTMLResponse(
                content=generate_help(),
                headers=_dap2_headers('dds'),
            )

        return router
