from pathlib import Path

from repodoctor.schemas import Edit
from repodoctor.tools.patch import apply_edits


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode())
    return p


def test_replaces_text_once(tmp_path):
    f = write(tmp_path, "a.py", "x = 1\ny = 2\n")
    r = apply_edits(tmp_path, [Edit(path="a.py", old="x = 1", new="x = 10")])
    assert r.ok and r.changed_files == ["a.py"]
    assert f.read_text() == "x = 10\ny = 2\n"


def test_text_not_found_leaves_file_unchanged(tmp_path):
    f = write(tmp_path, "a.py", "x = 1\n")
    r = apply_edits(tmp_path, [Edit(path="a.py", old="zzz", new="q")])
    assert not r.ok and "not found" in r.errors[0]
    assert f.read_text() == "x = 1\n"


def test_ambiguous_match_is_rejected(tmp_path):
    f = write(tmp_path, "a.py", "x = 1\nx = 1\n")
    r = apply_edits(tmp_path, [Edit(path="a.py", old="x = 1", new="x = 2")])
    assert not r.ok and "2 times" in r.errors[0]
    assert f.read_text() == "x = 1\nx = 1\n"


def test_atomic_when_a_later_edit_fails(tmp_path):
    f = write(tmp_path, "a.py", "x = 1\n")
    r = apply_edits(
        tmp_path,
        [Edit(path="a.py", old="x = 1", new="x = 2"), Edit(path="a.py", old="nope", new="y")],
    )
    assert not r.ok
    assert f.read_text() == "x = 1\n"


def test_sequential_edits_on_same_file(tmp_path):
    f = write(tmp_path, "a.py", "a\nb\n")
    r = apply_edits(
        tmp_path,
        [Edit(path="a.py", old="a", new="b2"), Edit(path="a.py", old="b2\nb", new="done")],
    )
    assert r.ok and f.read_text() == "done\n"


def test_creates_new_file_in_new_directory(tmp_path):
    r = apply_edits(tmp_path, [Edit(path="tests/test_new.py", old="", new="def test_x(): ...\n")])
    assert r.ok
    assert (tmp_path / "tests" / "test_new.py").read_text() == "def test_x(): ...\n"


def test_empty_old_on_existing_file_is_rejected(tmp_path):
    write(tmp_path, "a.py", "x\n")
    r = apply_edits(tmp_path, [Edit(path="a.py", old="", new="y")])
    assert not r.ok and "already exists" in r.errors[0]


def test_missing_file_is_rejected(tmp_path):
    r = apply_edits(tmp_path, [Edit(path="nope.py", old="a", new="b")])
    assert not r.ok and "does not exist" in r.errors[0]


def test_path_traversal_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    r = apply_edits(root, [Edit(path="../evil.txt", old="", new="x")])
    assert not r.ok
    assert not (tmp_path / "evil.txt").exists()


def test_absolute_path_outside_root_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    r = apply_edits(root, [Edit(path=str(outside), old="", new="x")])
    assert not r.ok and not outside.exists()


def test_git_directory_is_protected(tmp_path):
    r = apply_edits(tmp_path, [Edit(path=".git/config", old="", new="x")])
    assert not r.ok


def test_crlf_files_match_lf_text_and_keep_crlf(tmp_path):
    f = write(tmp_path, "a.txt", "a\r\nb\r\n")
    r = apply_edits(tmp_path, [Edit(path="a.txt", old="a\nb", new="x\ny")])
    assert r.ok
    assert f.read_bytes() == b"x\r\ny\r\n"