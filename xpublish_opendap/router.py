"""
xpublish_opendap

OpenDAP router for Xpublish
"""
import logging

import cachey
import opendap_protocol as dap
import xarray as xr
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from xpublish.dependencies import get_cache, get_dataset

from .dap_xarray import dap_dataset

logger = logging.getLogger("uvicorn")


dap_router = APIRouter()


def get_dap_dataset(
    dataset_id: str,
    ds: xr.Dataset = Depends(get_dataset),
    cache: cachey.Cache = Depends(get_cache),
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


@dap_router.get(".dds")
def dds_response(request: Request, dataset: dap.Dataset = Depends(get_dap_dataset)):
    """OpenDAP DDS response (types and dimension metadata)"""
    constraint = request.url.components[3]

    return StreamingResponse(
        dataset.dds(constraint=constraint),
        media_type="text/plain",
    )


@dap_router.get(".das")
def das_response(request: Request, dataset: dap.Dataset = Depends(get_dap_dataset)):
    """OpenDAP DAS response (attribute metadata)"""
    constraint = request.url.components[3]

    return StreamingResponse(
        dataset.das(constraint=constraint),
        media_type="text/plain",
    )


@dap_router.get(".dods")
def dods_response(request: Request, dataset: dap.Dataset = Depends(get_dap_dataset)):
    """OpenDAP dods response (data access)"""
    constraint = request.url.components[3]

    return StreamingResponse(
        dataset.dods(constraint=constraint),
        media_type="application/octet-stream",
    )
