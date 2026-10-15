import importlib

import pytest

import asymmetric_autoencoder


def test_package_has_version():
    assert asymmetric_autoencoder.__version__


@pytest.mark.parametrize("sub", ["config", "data", "models", "training", "evaluation", "utils"])
def test_subpackages_import(sub):
    importlib.import_module(f"asymmetric_autoencoder.{sub}")
