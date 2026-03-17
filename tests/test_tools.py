import pytest
from tools.executor import execute_code
from tools.searcher import search_error
from tools.patcher import patch_code


def test_execute_code_success():
    result = execute_code("print('hello')")
    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["stderr"] == ""
    assert result["timeout"] is False


def test_execute_code_error():
    result = execute_code("1/0")
    assert result["success"] is False
    assert "ZeroDivisionError" in result["stderr"]
    assert result["timeout"] is False


def test_execute_code_timeout():
    result = execute_code("import time; time.sleep(10)", timeout=1)
    assert result["timeout"] is True


def test_search_error_index_error():
    suggestion = search_error("IndexError: list index out of range")
    assert "index" in suggestion.lower() or "range" in suggestion.lower()


def test_patch_code_invalid_returns_original():
    original = "print('hello')"
    bad_patch = "def foo(:"
    result = patch_code(original, bad_patch)
    assert result == original
