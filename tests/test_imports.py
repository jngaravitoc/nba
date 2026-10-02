import importlib

import pytest

SUBPACKAGES = ["com", "cosmology", "ios", "kinematics", "orbits", "structure", "utils", "visuals"]


def test_version():
    import nba

    assert isinstance(nba.__version__, str)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"nba.{name}")
