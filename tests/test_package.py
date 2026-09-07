"""Tests for package-level attributes and functionality.

This module contains tests for package-level attributes like __version__
and other package-wide functionality.
"""

import re

import pytest
from packaging.version import InvalidVersion, Version

import ifunnel


def test_version():
    """Test that the __version__ attribute exists and has the expected format.

    This test verifies that:
    1. The __version__ attribute is defined in the package
    2. The version is a valid PEP 440 version whose release segment is X.Y.Z

    The full string is deliberately *not* required to be a bare ``X.Y.Z``. Since
    ``[project].version`` became dynamic (derived from the git tag by hatch-vcs), only
    a build made exactly on a tag reads ``0.1.0``; every other commit reads a PEP 440
    development version such as ``0.1.1.dev249+gb4afa5e2b.d20260907``. Asserting the
    bare form here would mean this test only passed on release builds -- so the
    invariant is checked where it still holds: the string must parse as PEP 440, and
    its release segment must have the three components semantic versioning requires.

    ``packaging`` is a pytest dependency, so it is available wherever this test runs.
    """
    assert hasattr(ifunnel, "__version__")
    assert isinstance(ifunnel.__version__, str)

    try:
        version = Version(ifunnel.__version__)
    except InvalidVersion:
        pytest.fail(f"__version__ {ifunnel.__version__!r} is not a valid PEP 440 version")

    # The release segment (what `base_version` returns) must still be X.Y.Z, matching
    # the vX.Y.Z tags the version is derived from.
    assert re.match(r"^\d+\.\d+\.\d+$", version.base_version) is not None, (
        f"__version__ {ifunnel.__version__!r} has release segment {version.base_version!r}, which is not X.Y.Z"
    )
