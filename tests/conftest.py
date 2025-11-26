import pytest
from nvxpy.variable import reset_variable_ids


@pytest.fixture(autouse=True)
def clear_variable_names():
    """Automatically reset variable IDs between each test"""
    reset_variable_ids()
    yield