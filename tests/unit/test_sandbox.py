from repodoctor.tools.repo import ProjectInfo
from repodoctor.tools.sandbox import build_install_commands, truncate_output


def test_truncate_small_output_unchanged():
    assert truncate_output("a\nb\nc") == "a\nb\nc"


def test_truncate_keeps_the_tail_and_says_so():
    text = "\n".join(f"line{i}" for i in range(1000))
    out = truncate_output(text, max_lines=5)
    assert "line999" in out and "line0\n" not in out
    assert "995 earlier lines truncated" in out


def test_truncate_caps_characters():
    out = truncate_output("x" * 100_000, max_chars=100)
    assert len(out) < 200 and "truncated" in out


def test_install_commands_full(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='p'\n")
    info = ProjectInfo(
        True, extras=["tests"], dependency_groups=["dev"], requirement_files=["requirements.txt"]
    )
    assert build_install_commands(tmp_path, info) == [
        "pip install -r requirements.txt",
        "pip install -e '.[tests]'",
        "pip install --group dev",
        "pip install pytest",
    ]


def test_install_commands_without_extras(tmp_path):
    (tmp_path / "setup.py").write_text("")
    cmds = build_install_commands(tmp_path, ProjectInfo(True))
    assert cmds == ["pip install -e .", "pip install pytest"]


def test_install_commands_requirements_only(tmp_path):
    info = ProjectInfo(True, requirement_files=["requirements.txt"])
    assert build_install_commands(tmp_path, info) == [
        "pip install -r requirements.txt",
        "pip install pytest",
    ]