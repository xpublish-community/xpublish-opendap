## xpublish_opendap

[![Tests](https://github.com/gulfofmaine/xpublish-opendap/actions/workflows/tests.yml/badge.svg)](https://github.com/gulfofmaine/xpublish-opendap/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/xpublish-community/xpublish-opendap/branch/main/graph/badge.svg?token=0HMS1Q8Z8Y)](https://codecov.io/gh/xpublish-community/xpublish-opendap)

Quick description

### Documentation and code

URLs for the docs and code.

### Installation - not really, install from Github instead

For `conda` users you can

```shell
conda install --channel conda-forge xpublish_opendap
```

or, if you are a `pip` users

```shell
pip install xpublish_opendap
```

### Example

```python
import xarray as xr
import xpublish
from xpublish.routers import base_router, zarr_router
from xpublish_opendap import dap_router


ds = xr.open_dataset("dataset.nc")

rest = xpublish.Rest(
    ds,
    routers=[
        (base_router, {"tags": ["info"]}),
        (dap_router, {"tags": ["opendap"], "prefix": "/dap"}),
        (zarr_router, {"tags": ["zarr"], "prefix": "/zarr"})
    ]
)
```

## Get in touch

Report bugs, suggest features or view the source code on [GitHub](https://github.com/ioos/xpublish_opendap/issues).

## License and copyright

xpublish_opendap is licensed under BSD 3-Clause "New" or "Revised" License (BSD-3-Clause).

Development occurs on GitHub at <https://github.com/ioos/xpublish_opendap>.
