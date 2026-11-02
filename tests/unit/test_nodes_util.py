from repodoctor.nodes.util import render_files


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