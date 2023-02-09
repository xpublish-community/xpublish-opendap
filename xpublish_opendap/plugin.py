"""
xpublish_opendap

OpenDAP router for Xpublish
"""
import logging
from typing import List
from urllib import parse

import cachey
import opendap_protocol as dap
import xarray as xr
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from xpublish import Plugin, hookimpl, Dependencies

from .dap_xarray import dap_dataset

logger = logging.getLogger("uvicorn")



class OpenDapPlugin(Plugin):
    name = "opendap"

    dataset_router_prefix = "/opendap"
    dataset_router_tags: List[str] = ["opendap"]

    @hookimpl
    def dataset_router(self, deps: Dependencies):
        router = APIRouter(prefix=self.dataset_router_prefix, tags=self.dataset_router_tags)

        def get_dap_dataset(
            dataset_id: str,
            ds: xr.Dataset = Depends(deps.dataset),
            cache: cachey.Cache = Depends(deps.cache),
        ):
            """
            Get a dataset that has been translated to opendap
            """
            cache_key = f"opendap_dataset_{dataset_id}"
            dataset = cache.get(cache_key)

            if dataset is None:
                dataset = dap_dataset(ds, dataset_id)

                cache.put(cache_key, dataset, 99999)

            return dataset

        def dap_constraint(request: Request) -> str:
            """Parse DAP constraints from request"""
            constraint = parse.unquote(request.url.components[3])

            return constraint

        @router.get(".dds")
        def dds_response(
            constraint=Depends(dap_constraint), dataset: dap.Dataset = Depends(get_dap_dataset),
        ):
            """OpenDAP DDS response (types and dimension metadata)"""
            return StreamingResponse(
                dataset.dds(constraint=constraint), media_type="text/plain",
            )

        @router.get(".das")
        def das_response(
            constraint=Depends(dap_constraint), dataset: dap.Dataset = Depends(get_dap_dataset),
        ):
            """OpenDAP DAS response (attribute metadata)"""
            return StreamingResponse(
                dataset.das(constraint=constraint), media_type="text/plain",
            )


        @router.get(".dods")
        def dods_response(
            constraint=Depends(dap_constraint), dataset: dap.Dataset = Depends(get_dap_dataset),
        ):
            """OpenDAP dods response (data access)"""
            return StreamingResponse(
                dataset.dods(constraint=constraint), media_type="application/octet-stream",
            )

        return router
