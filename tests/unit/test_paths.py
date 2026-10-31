from repodoctor.tools.paths import safe_resolve


def test_inside_root_is_allowed(tmp_path):
    assert safe_resolve(tmp_path, "a/b.py") == tmp_path.resolve() / "a" / "b.py"


def test_root_itself_is_allowed(tmp_path):
    assert safe_resolve(tmp_path, ".") == tmp_path.resolve()


def test_parent_escape_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert safe_resolve(root, "../x") is None


def test_absolute_path_outside_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert safe_resolve(root, str(tmp_path / "other.txt")) is None


def test_git_dir_is_rejected(tmp_path):
    assert safe_resolve(tmp_path, ".git/config") is None
    assert safe_resolve(tmp_path, ".git") is None