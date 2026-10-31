import pytest

from repodoctor.tools.search import (
    MAX_FILE_BYTES,
    MAX_HIT_CHARS,
    GrepHit,
    GrepResult,
    SearchError,
    find_symbol,
    format_grep,
    grep,
    list_dir,
    read_file,
    read_raw,
)

APP = (
    "import os\n"                       # 1
    "\n"                                # 2
    "def add(a, b):\n"                  # 3
    "    return a + b\n"                # 4
    "\n"                                # 5
    "async def fetch(url):\n"           # 6
    "    return url\n"                  # 7
    "\n"                                # 8
    "class Calc:\n"                     # 9
    "    def add(self, x):\n"           # 10
    "        return x\n"                # 11
    "\n"                                # 12
    "def add_all(items):\n"             # 13
    "    return sum(items)\n"           # 14
)

FILES = {
    "app.py": APP,
    "pkg/util.py": "def helper():\n    return 'a.b(c)'\n",
    "README.md": "def add is documented here\n",
    "data.bin": b"\x00\x01add\x02",
    "empty.txt": "",
    "crlf.txt": b"a\r\nb\r\n",
    ".gitignore": "ignored.txt\n",
    "ignored.txt": "secret needle\n",
    "kept.txt": "needle\n",
}


@pytest.fixture
def root(make_git_repo):
    return make_git_repo(FILES)


# ---------- grep ----------

def test_grep_finds_matches_with_line_numbers(root):
    r = grep(root, "def helper")
    assert [(h.path, h.line) for h in r.hits] == [("pkg/util.py", 1)]
    assert r.hits[0].text == "def helper():"


def test_grep_skips_binary_files(root):
    assert "data.bin" not in {h.path for h in grep(root, "add").hits}


def test_grep_fixed_string_vs_regex(root):
    assert [h.path for h in grep(root, "a.b(c)", fixed=True).hits] == ["pkg/util.py"]
    assert grep(root, "a.b(c)").hits == []  # as a regex, the parentheses are a group


def test_grep_regex(root):
    r = grep(root, r"^(async )?def [a-z_]+", path="app.py")
    assert [h.line for h in r.hits] == [3, 6, 13]


def test_grep_ignore_case(root):
    assert [h.line for h in grep(root, "CALC", ignore_case=True).hits] == [9]


def test_grep_no_match_is_empty_not_error(root):
    r = grep(root, "zzz_nothing_zzz")
    assert r.hits == [] and not r.truncated


def test_grep_invalid_regex_raises(root):
    with pytest.raises(SearchError, match="search failed"):
        grep(root, "(unclosed")


def test_grep_empty_pattern_raises(root):
    with pytest.raises(SearchError):
        grep(root, "")


def test_grep_truncates_results(make_git_repo):
    r = make_git_repo({"x.txt": "x\n" * 10}, name="many")
    res = grep(r, "x", max_results=3)
    assert len(res.hits) == 3 and res.truncated


def test_grep_scoped_to_path(root):
    assert {h.path for h in grep(root, "def", path="pkg").hits} == {"pkg/util.py"}


def test_grep_finds_untracked_files(root):
    (root / "new_test.py").write_text("def test_new(): ...\n")
    assert "new_test.py" in {h.path for h in grep(root, "test_new").hits}


def test_grep_honors_gitignore(root):
    assert {h.path for h in grep(root, "needle").hits} == {"kept.txt"}


@pytest.mark.parametrize("bad", ["../x", ".git", "nope/missing"])
def test_grep_rejects_bad_paths(root, bad):
    with pytest.raises(SearchError):
        grep(root, "x", path=bad)


def test_grep_truncates_long_lines(make_git_repo):
    r = make_git_repo({"long.txt": "needle" + "x" * 500 + "\n"}, name="long")
    (hit,) = grep(r, "needle").hits
    assert len(hit.text) <= MAX_HIT_CHARS + 1


# ---------- find_symbol ----------

def test_find_function_and_method_definitions(root):
    r = find_symbol(root, "add")
    assert {(h.path, h.line) for h in r.hits} == {("app.py", 3), ("app.py", 10)}


def test_find_async_function(root):
    assert [(h.path, h.line) for h in find_symbol(root, "fetch").hits] == [("app.py", 6)]


def test_find_class(root):
    assert [(h.path, h.line) for h in find_symbol(root, "Calc").hits] == [("app.py", 9)]


def test_find_symbol_is_exact_not_prefix(root):
    assert [h.line for h in find_symbol(root, "add_all").hits] == [13]


def test_find_symbol_only_looks_in_python_files(root):
    assert "README.md" not in {h.path for h in find_symbol(root, "add").hits}


@pytest.mark.parametrize("bad", ["", "a b", "x;y", "1abc", "a.b"])
def test_find_symbol_rejects_non_identifiers(root, bad):
    with pytest.raises(SearchError):
        find_symbol(root, bad)


# ---------- format_grep ----------

def test_format_grep_no_matches():
    assert format_grep(GrepResult()) == "No matches."


def test_format_grep_lists_hits():
    r = GrepResult(hits=[GrepHit("a.py", 3, "def f():"), GrepHit("b.py", 7, "x = 1")])
    assert format_grep(r) == "a.py:3: def f():\nb.py:7: x = 1"


def test_format_grep_notes_truncation():
    r = GrepResult(hits=[GrepHit("a.py", 1, "x")], truncated=True)
    out = format_grep(r)
    assert out.startswith("a.py:1: x") and "more matches" in out


# ---------- read_file ----------

def test_read_file_numbers_lines_and_range_header(root):
    out = read_file(root, "app.py", start=3, end=4)
    assert out.splitlines()[0] == "# app.py (lines 3-4 of 14)"
    assert "3| def add(a, b):" in out
    assert "4|     return a + b" in out


def test_read_file_whole_file(root):
    out = read_file(root, "app.py")
    assert " 1| import os" in out
    assert "14|     return sum(items)" in out


def test_read_file_end_is_clamped(root):
    assert "(lines 1-14 of 14)" in read_file(root, "app.py", end=999)


def test_read_file_truncates_with_continuation_hint(root):
    out = read_file(root, "app.py", max_lines=5)
    assert "(lines 1-5 of 14)" in out and "start=6" in out


def test_read_file_start_past_end_raises(root):
    with pytest.raises(SearchError, match="past the end"):
        read_file(root, "app.py", start=99)


def test_read_file_end_before_start_raises(root):
    with pytest.raises(SearchError):
        read_file(root, "app.py", start=5, end=2)


@pytest.mark.parametrize("bad", ["missing.py", "pkg", "../x", ".git/config"])
def test_read_file_rejects_bad_paths(root, bad):
    with pytest.raises(SearchError):
        read_file(root, bad)


def test_read_file_rejects_binary(root):
    with pytest.raises(SearchError, match="binary"):
        read_file(root, "data.bin")


def test_read_file_empty(root):
    assert "(empty file)" in read_file(root, "empty.txt")


def test_read_file_handles_crlf(root):
    out = read_file(root, "crlf.txt")
    assert "of 2)" in out and "\r" not in out


def test_read_file_rejects_huge_files(root):
    (root / "big.txt").write_text("x" * (MAX_FILE_BYTES + 1))
    with pytest.raises(SearchError, match="too large"):
        read_file(root, "big.txt")


# ---------- list_dir ----------

def test_list_dir_dirs_first_and_hides_git(root):
    names = list_dir(root).splitlines()
    assert names[0] == "pkg/"
    assert "app.py" in names and ".git/" not in names and ".git" not in names


def test_list_dir_skips_caches(root):
    (root / "__pycache__").mkdir()
    assert "__pycache__/" not in list_dir(root).splitlines()


def test_list_dir_subdirectory(root):
    assert list_dir(root, "pkg") == "util.py"


def test_list_dir_rejects_a_file(root):
    with pytest.raises(SearchError, match="not a directory"):
        list_dir(root, "app.py")


def test_list_dir_empty_directory(root):
    (root / "emptydir").mkdir()
    assert list_dir(root, "emptydir") == "(empty directory)"


def test_list_dir_truncates(root):
    out = list_dir(root, max_entries=2)
    assert "more entries not shown" in out

def test_read_raw_returns_plain_text_without_numbers(root):
    assert read_raw(root, "app.py") == APP


def test_read_raw_normalizes_crlf(root):
    assert read_raw(root, "crlf.txt") == "a\nb\n"


def test_read_raw_rejects_binary(root):
    with pytest.raises(SearchError, match="binary"):
        read_raw(root, "data.bin")