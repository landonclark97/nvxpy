import pytest
from nvxpy.variable import Variable

@pytest.fixture(autouse=True)
def clear_variable_names():
    """Automatically clear variable names between each test"""
    Variable._used_names.clear()
    Variable._ids = 0
    yield 