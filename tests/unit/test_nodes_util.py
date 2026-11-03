from repodoctor.nodes.util import existing_files, render_files


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
