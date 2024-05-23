"""Test against the same matrix as Github Actions."""

import nox
import yaml

with open("./.github/workflows/tests.yml") as f:
    workflow = yaml.safe_load(f)

python_versions = workflow["jobs"]["run"]["strategy"]["matrix"]["python-version"]
pydantic_versions = workflow["jobs"]["run"]["strategy"]["matrix"]["pydantic-version"]

nox.options.default_venv_backend = "uv"


@nox.session(python=python_versions)
@nox.parametrize("pydantic", pydantic_versions)
def tests(session: nox.Session, pydantic: str):
    """Run py.test against Github Actions matrix.

    Allows passing additional arguments to py.test with with postargs,
    for example: `nox -- --pdb`.
    """
    session.install("-r", "requirements-dev.txt")
    session.install(".")
    session.install(f"pydantic{pydantic}")
    session.run("pytest", "--verbose", *session.posargs)
