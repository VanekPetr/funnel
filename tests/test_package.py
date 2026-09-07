"""Tests for package-level attributes and functionality.

This module contains tests for package-level attributes like __version__
and other package-wide functionality.
"""

import pytest
from packaging.version import InvalidVersion, Version

import ifunnel


def test_version():
    """Test that the __version__ attribute exists and is a valid PEP 440 version.

    ``[project].version`` is dynamic — hatch-vcs derives it from ``git describe`` — so
    the string this reads depends on the *shape of the clone*, not just on the code:

    * a build made exactly on a tag reads ``0.1.0``;
    * a commit past the tag reads ``0.1.1.dev249+gb4afa5e2b``;
    * a **shallow** clone that cannot see any tag reads ``0.1.dev1+gdfaa814a2`` —
      note the two-component ``0.1`` release segment.

    That last case is what CI produces: ``rhiza_ci.yml`` checks out at the default
    depth, so requiring three components (or a bare ``X.Y.Z``) fails there for a reason
    that has nothing to do with the package being correct.

    So the assertion is what remains true in every one of those cases: ``__version__``
    is present, is a string, and parses as PEP 440. The three-component release segment
    is enforced where it is actually meaningful — on the built distribution, by
    ``rhiza_release.yml``'s "Verify built distribution matches the tag" step, which is
    the only place a wrong version can still do damage.

    ``packaging`` is a pytest dependency, so it is available wherever this test runs.
    """
    assert hasattr(ifunnel, "__version__")
    assert isinstance(ifunnel.__version__, str)
    assert ifunnel.__version__, "__version__ is empty"

    try:
        Version(ifunnel.__version__)
    except InvalidVersion:
        pytest.fail(f"__version__ {ifunnel.__version__!r} is not a valid PEP 440 version")
