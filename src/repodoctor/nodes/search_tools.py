"""LangChain wrappers around tools/search.py for the explorer agent."""

from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from repodoctor.tools import search


def _safe(fn: Callable[[], str]) -> str:
    try:
        return fn()
    except search.SearchError as e:
        return f"ERROR: {e}"  # returned to the model so it can correct itself


def make_search_tools(root: Path) -> list[BaseTool]:
    @tool
    def grep_code(
        pattern: str, path: str = ".", fixed_string: bool = False, ignore_case: bool = False
    ) -> str:
        """Search the repository's text files. `pattern` is an extended regular expression
        (or a literal string when fixed_string is true). Returns 'file:line: text' lines.
        Limit the search with `path` (a file or directory)."""
        return _safe(
            lambda: search.format_grep(
                search.grep(
                    root, pattern, path=path, fixed=fixed_string, ignore_case=ignore_case
                )
            )
        )

    @tool
    def find_definition(name: str) -> str:
        """Find where a Python function, method or class is defined, by its exact name."""
        return _safe(lambda: search.format_grep(search.find_symbol(root, name)))

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file with line numbers. The 'N|' prefix is not part of the file content.
        end_line 0 means read up to 400 lines starting at start_line."""
        return _safe(
            lambda: search.read_file(
                root, path, start=start_line, end=end_line if end_line > 0 else None
            )
        )

    @tool
    def list_dir(path: str = ".") -> str:
        """List a directory. Directories are shown with a trailing slash."""
        return _safe(lambda: search.list_dir(root, path))

    return [grep_code, find_definition, read_file, list_dir]