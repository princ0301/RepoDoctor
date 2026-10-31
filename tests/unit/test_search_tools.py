import pytest

from repodoctor.nodes.search_tools import make_search_tools


@pytest.fixture
def tools(make_git_repo):
    root = make_git_repo({"app.py": "def add(a, b):\n    return a + b\n"})
    return {t.name: t for t in make_search_tools(root)}


def test_tool_names(tools):
    assert set(tools) == {"grep_code", "find_definition", "read_file", "list_dir"}


def test_find_definition(tools):
    assert tools["find_definition"].invoke({"name": "add"}) == "app.py:1: def add(a, b):"


def test_read_file_with_default_arguments(tools):
    out = tools["read_file"].invoke({"path": "app.py"})
    assert "1| def add(a, b):" in out


def test_errors_are_returned_not_raised(tools):
    assert tools["read_file"].invoke({"path": "missing.py"}).startswith("ERROR:")
    assert tools["grep_code"].invoke({"pattern": "(bad"}).startswith("ERROR:")