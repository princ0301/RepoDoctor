import pytest

from repodoctor.tools.diffcheck import (
    check_diff,
    parse_diff,
    repo_line_length,
    syntax_findings,
    unused_import_findings,
)

HEAD = [
    "diff --git a/src/a.py b/src/a.py",
    "index 111..222 100644",
    "--- a/src/a.py",
    "+++ b/src/a.py",
    "@@ -1,3 +1,4 @@",
    " import os",
    "-x = 1",
]
TAIL = [" z = 3"]


def diff(*added: str, head=HEAD) -> str:
    return "\n".join([*head, *added, *TAIL]) + "\n"


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


def test_parse_tracks_files_line_numbers_and_removals():
    p = parse_diff(diff("+x = 2", "+y = 3"))
    assert p.files == ["src/a.py"] and p.n_removed == 1
    assert [(a.line, a.text) for a in p.added] == [(2, "x = 2"), (3, "y = 3")]


def test_parse_new_and_deleted_files():
    d = "\n".join([
        "diff --git a/tests/test_new.py b/tests/test_new.py", "new file mode 100644",
        "--- /dev/null", "+++ b/tests/test_new.py", "@@ -0,0 +1,2 @@",
        "+def test_a():", "+    assert True",
        "diff --git a/old.py b/old.py", "deleted file mode 100644",
        "--- a/old.py", "+++ /dev/null", "@@ -1 +0,0 @@", "-gone = 1",
    ]) + "\n"
    p = parse_diff(d)
    assert p.files == ["tests/test_new.py", "old.py"] and p.deleted == ["old.py"]
    assert [(a.line, a.text) for a in p.added] == [(1, "def test_a():"), (2, "    assert True")]


def test_a_missing_final_newline_is_found_on_the_added_line():
    d = "\n".join([*HEAD, "+x = 2", "\\ No newline at end of file"]) + "\n"
    assert [a.text for a in parse_diff(d).no_eol] == ["x = 2"]
    assert "no_newline_at_eof" in codes(check_diff(d))


def test_a_clean_diff_has_no_findings():
    assert check_diff(diff("+x = 2")).findings == []


def test_style_findings_for_python_lines():
    r = check_diff(diff("+x = 2  ", "+y = '\tv'", "+z = " + "a" * 100), line_length=88)
    assert {"trailing_whitespace", "tab_character", "long_lines"} <= codes(r)


def test_crlf_is_reported_but_is_not_trailing_whitespace():
    r = check_diff(diff("+x = 2\r"))
    assert "crlf" in codes(r) and "trailing_whitespace" not in codes(r)


def test_secrets_block_the_change():
    token = "ghp_" + "a" * 36
    assert check_diff(diff(f"+TOKEN = '{token}'")).verdict == "skip"
    mine = check_diff(diff("+KEY = 'supersecretvalue'"), secrets=("supersecretvalue",))
    assert mine.verdict == "skip" and "secret_in_diff" in codes(mine)


def test_ci_files_block_the_change():
    head = [
        "diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml",
        "--- a/.github/workflows/ci.yml", "+++ b/.github/workflows/ci.yml", "@@ -1 +1 @@",
    ]
    assert "touches_ci" in codes(check_diff("\n".join([*head, "+on: push"]) + "\n"))


def test_size_packaging_deletion_and_test_name_warnings():
    d = "\n".join([
        "diff --git a/pyproject.toml b/pyproject.toml", "--- a/pyproject.toml",
        "+++ b/pyproject.toml", "@@ -1 +1 @@", "+a = 1",
        "diff --git a/tests/test_repodoctor_repro.py b/tests/test_repodoctor_repro.py",
        "new file mode 100644", "--- /dev/null", "+++ b/tests/test_repodoctor_repro.py",
        "@@ -0,0 +1 @@", "+def test_a(): ...",
        "diff --git a/old.py b/old.py", "deleted file mode 100644", "--- a/old.py",
        "+++ /dev/null", "@@ -1 +0,0 @@", "-x = 1",
    ]) + "\n"
    r = check_diff(d, max_files=2, max_lines=2)
    assert {"touches_packaging", "generated_test_name", "deletes_files", "many_files",
            "large_change"} <= codes(r)
    assert r.verdict == "warn"


def test_an_empty_diff_is_blocked():
    assert check_diff("").verdict == "skip"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("pyproject.toml", "[tool.ruff]\nline-length = 100\n", 100),
        ("pyproject.toml", "[tool.black]\nline-length = 79\n", 79),
        ("setup.cfg", "[flake8]\nmax-line-length = 120\n", 120),
        ("README.md", "nothing\n", 88),
    ],
)
def test_the_projects_own_line_length_is_used(tmp_path, name, text, expected):
    (tmp_path / name).write_text(text)
    assert repo_line_length(tmp_path) == expected


def test_syntax_errors_are_reported(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("def f(:\n")
    found = syntax_findings(tmp_path, ["ok.py", "bad.py", "missing.py", "notes.md"])
    assert [f.code for f in found] == ["syntax_error"] and "bad.py" in found[0].message


def test_only_unused_imports_added_by_the_change_are_reported(tmp_path):
    (tmp_path / "t.py").write_text(
        "import os\nimport pytest\n\nimport sqlparse\n\n\ndef test_a():\n    sqlparse.split('a')\n"
    )
    d = "\n".join([
        "diff --git a/t.py b/t.py", "--- a/t.py", "+++ b/t.py", "@@ -1,2 +1,3 @@",
        " import os", "+import pytest", " ", "+import sqlparse",
    ]) + "\n"
    found = unused_import_findings(tmp_path, d)
    assert [f.message for f in found] == ["t.py:2: 'pytest' is imported but not used."]
