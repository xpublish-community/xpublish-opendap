"""Test OpenDAP server with air temperature dataset."""

import xpublish
from datasets import datasets

from xpublish_opendap import OpenDapPlugin

rest = xpublish.Rest(
    datasets,
    plugins={"opendap": OpenDapPlugin()},
)

rest.serve()
