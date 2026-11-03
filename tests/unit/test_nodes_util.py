from repodoctor.nodes.util import existing_files, render_files, repo_file_list, test_file_of, similar_sites


def test_repo_file_list_puts_source_before_tests(make_git_repo):
    root = make_git_repo({"pkg/a.py": "x\n", "tests/test_a.py": "x\n", "README.md": "x\n"})
    assert repo_file_list(root) == "pkg/a.py\ntests/test_a.py"


def test_repo_file_list_is_capped(make_git_repo):
    root = make_git_repo({f"m{i}.py": "x\n" for i in range(10)})
    out = repo_file_list(root, max_files=3)
    assert "m0.py" in out and "m3.py" not in out and "7 more" in out


def test_existing_files_accepts_a_unique_suffix(make_git_repo):
    root = make_git_repo({"src/pkg/core.py": "x\n", "src/pkg/other.py": "y\n"})
    assert existing_files(root, ["pkg/core.py", "core.py"]) == ["src/pkg/core.py"]


def test_existing_files_rejects_an_ambiguous_suffix(make_git_repo):
    root = make_git_repo({"a/util.py": "x\n", "b/util.py": "y\n"})
    assert existing_files(root, ["util.py"]) == []


def test_existing_files_keeps_exact_paths_and_drops_unknown_or_unsafe(make_git_repo):
    root = make_git_repo({"src/a.py": "x\n"})
    assert existing_files(root, ["./src/a.py", "nope.py", "../x.py", "/etc/passwd"]) == [
        "src/a.py"
    ]


def test_render_files_marks_what_the_budget_cut(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n" * 20)
    (tmp_path / "b.py").write_text("b = 2\n" * 20)
    text, omitted = render_files(tmp_path, ["a.py", "b.py"], budget=150)
    assert omitted == ["b.py"]
    assert "a = 1" in text and "b = 2" not in text and "NOT SHOWN" in text


def test_render_files_always_shows_the_first_file(tmp_path):
    (tmp_path / "big.py").write_text("x = 1\n" * 1000)
    text, omitted = render_files(tmp_path, ["big.py"], budget=10)
    assert omitted == [] and "x = 1" in text


def test_render_files_reports_unavailable_files(tmp_path):
    text, omitted = render_files(tmp_path, ["missing.py"], budget=100)
    assert "unavailable" in text and omitted == []

def test_test_file_of_maps_a_junit_id_to_its_file(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("x\n")
    assert test_file_of(tmp_path, "tests.test_a.TestA::test_m[1]") == "tests/test_a.py"
    assert test_file_of(tmp_path, "tests.test_a::test_m") == "tests/test_a.py"
    assert test_file_of(tmp_path, "other.mod::test_m") is None

def test_similar_sites_finds_matching_lines(make_git_repo):
    root = make_git_repo({
        "a.py": "def f(self):\n    for field in self._field_names:\n        x = 1\n"
                "    for field in self._field_names:\n        y = 2\n"
    })
    assert similar_sites(root, r"for field in self\._field_names", ["a.py"]) == [
        "a.py:2: for field in self._field_names:",
        "a.py:4: for field in self._field_names:",
    ]


def test_similar_sites_ignores_empty_bad_or_too_broad_patterns(make_git_repo):
    root = make_git_repo({"a.py": "x = 1\n" * 40})
    assert similar_sites(root, "", ["a.py"]) == []
    assert similar_sites(root, "(unclosed", ["a.py"]) == []
    assert similar_sites(root, "x", ["a.py"], limit=5) == []
