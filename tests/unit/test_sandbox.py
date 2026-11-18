from repodoctor.tools.repo import ProjectInfo
from repodoctor.tools.sandbox import (
    ExecResult,
    Sandbox,
    build_install_commands,
    clip_failure_output,
    truncate_output,
)


def test_truncate_small_output_unchanged():
    assert truncate_output("a\nb\nc") == "a\nb\nc"


def test_truncate_keeps_the_tail_and_says_so():
    text = "\n".join(f"line{i}" for i in range(1000))
    out = truncate_output(text, max_lines=5)
    assert "line999" in out and "line0" in out
    assert "lines omitted" in out


def test_truncate_caps_characters():
    out = truncate_output("x" * 100_000, max_chars=100)
    assert len(out) < 200 and "omitted" in out


def test_clipping_keeps_the_assertion_lines_pytest_prints_before_captured_output():
    head = [
        "___ test_x ___",
        "",
        "    def test_x():",
        ">       assert foo() == 2",
        "E       assert 1 == 2",
    ]
    captured = [f"captured line {i}" for i in range(300)]
    text = "\n".join([*head, "----- Captured stdout call -----", *captured, "1 failed in 1s"])
    out = clip_failure_output(text, 40, 10_000)
    assert "E       assert 1 == 2" in out and ">       assert foo() == 2" in out
    assert "1 failed in 1s" in out and "lines omitted" in out
    assert "captured line 150" not in out and len(out.splitlines()) <= 45
    assert clip_failure_output("a\nb\nc", 40, 1000) == "a\nb\nc"


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

def test_install_retries_only_after_a_timeout(tmp_path):
    calls: list[int] = []

    class Stalls(Sandbox):
        def run(self, cmd, timeout):
            calls.append(timeout)
            first = len(calls) == 1
            return ExecResult(-1 if first else 0, "", first, 0.0)

    assert Stalls(tmp_path).install(ProjectInfo(True), timeout=7, attempts=2).ok
    assert calls == [7, 7]

    calls.clear()
    assert not Stalls(tmp_path).install(ProjectInfo(True), timeout=7, attempts=1).ok
    assert calls == [7]


def test_install_exports_the_projects_version_variable(tmp_path):
    seen: list[str] = []

    class Spy(Sandbox):
        def run(self, cmd, timeout):
            seen.append(cmd)
            return ExecResult(0, "", False, 0.0)

    Spy(tmp_path).install(
        ProjectInfo(True, install_env={"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_X": "1.2.3"})
    )
    assert "export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_X=1.2.3; " in seen[0]
