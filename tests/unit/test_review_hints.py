from repodoctor.tools.review_hints import (
    changelog_findings,
    find_changelog,
    hint_new_test_files,
    suggest_test_homes,
    unknown_test_id,
)


def make_tests(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_split.py").write_text(
        "import sqlparse\n\n\ndef test_a():\n    sqlparse.split('a;')\n    sqlparse.split('b;')\n"
    )
    (tests / "test_format.py").write_text(
        "import sqlparse\n\n\ndef test_b():\n    sqlparse.format('a')\n"
    )
    (tests / "test_issue_9.py").write_text(
        "import sqlparse\n\n\ndef test_c():\n"
        "    parts = sqlparse.split('a; b')\n    assert str(parts[0]).strip() == 'a;'\n"
    )


def test_find_changelog_looks_in_the_root_and_docs(tmp_path):
    assert find_changelog(tmp_path) is None
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHANGES.rst").write_text("x")
    assert find_changelog(tmp_path) == "docs/CHANGES.rst"
    (tmp_path / "CHANGELOG").write_text("x")
    assert find_changelog(tmp_path) == "CHANGELOG"


def test_changelog_findings(tmp_path):
    (tmp_path / "CHANGELOG").write_text("x")
    assert [f.code for f in changelog_findings(tmp_path, ["a.py"])] == ["changelog_not_updated"]
    assert changelog_findings(tmp_path, ["CHANGELOG", "a.py"]) == []
    assert changelog_findings(tmp_path / "missing", ["a.py"]) == []


def test_suggest_test_homes_prefers_files_that_call_the_same_functions(tmp_path):
    make_tests(tmp_path)
    homes = suggest_test_homes(tmp_path, "tests/test_issue_9.py")
    assert homes[0][0] == "tests/test_split.py" and "split" in homes[0][1]
    assert all(rel != "tests/test_issue_9.py" for rel, _ in homes)


def test_hint_new_test_files_names_a_better_home(tmp_path):
    make_tests(tmp_path)
    (finding,) = hint_new_test_files(tmp_path, ["tests/test_issue_9.py"])
    assert finding.code == "new_test_file" and "tests/test_split.py" in finding.message
    assert hint_new_test_files(tmp_path, []) == []


def test_unknown_test_id_names_the_tests_that_do_exist(tmp_path):
    (tmp_path / "t.py").write_text(
        "def test_a():\n    pass\n\n\nclass TestB:\n    def test_c(self):\n        pass\n"
    )
    assert unknown_test_id(tmp_path, "t.py::test_a") is None
    assert unknown_test_id(tmp_path, "t.py::TestB::test_c[1]") is None
    assert unknown_test_id(tmp_path, "t.py") is None
    msg = unknown_test_id(tmp_path, "t.py::test_nope")
    assert "test_nope is not a test in t.py" in msg and "test_a" in msg
    assert unknown_test_id(tmp_path, "missing.py::test_a") == "missing.py does not exist."
