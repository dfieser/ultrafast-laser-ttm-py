"""Shared fixtures: loading the MATLAB golden fixtures from validation/fixtures."""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.io

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "validation", "fixtures")


def _to_python(obj):
    """Recursively convert scipy.io.loadmat mat_struct objects to dicts."""
    if isinstance(obj, scipy.io.matlab.mat_struct):
        return {name: _to_python(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [_to_python(el) for el in obj]
    return obj


def load_fixture(case_name: str) -> dict:
    """Load one golden fixture (.mat, authoritative full precision) as dicts."""
    path = os.path.join(FIXTURES_DIR, f"{case_name}.mat")
    if not os.path.exists(path):
        pytest.skip(f"golden fixture {case_name}.mat not present "
                    "(generate with validation/generate_fixtures.m)")
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return _to_python(data["fx"])


@pytest.fixture
def fixture_loader():
    return load_fixture
