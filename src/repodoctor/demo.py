"""Tiny repos with one seeded bug, for end-to-end runs."""

from dataclasses import dataclass
from pathlib import Path

from repodoctor.tools.repo import remove_dir, run_git

DEMO_TEST_ID = "tests/test_text.py::test_truncate_text_exactly_max_len_is_unchanged"

_TEXTKIT = '''\
"""Tiny text helpers used by the RepoDoctor demo."""


def word_count(text: str) -> int:
    return len(text.split())


def truncate(text: str, max_len: int, suffix: str = "...") -> str:
    """Shorten text to at most max_len characters, ending with suffix if it was cut."""
    if len(text) < max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def title_case(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())
'''

_TESTS_BLIND = '''\
from textkit import title_case, truncate, word_count


def test_word_count():
    assert word_count("a b  c") == 3


def test_title_case():
    assert title_case("hello big world") == "Hello Big World"


def test_truncate_short_text_unchanged():
    assert truncate("hi", 10) == "hi"


def test_truncate_long_text():
    assert truncate("hello world", 8) == "hello..."
'''

_EXTRA_TEST = '''

def test_truncate_text_exactly_max_len_is_unchanged():
    assert truncate("hello", 5) == "hello"
'''

_ISSUE = """\
# truncate() shortens text that already fits

`truncate("hello", 5)` returns `"he..."`, but it should return `"hello"`.
The text is exactly 5 characters long and max_len is 5, so nothing needs to be cut.
Text that fits within max_len should never be shortened.
"""


@dataclass
class Demo:
    repo_dir: Path          # has a failing test for the bug (use with --test)
    blind_repo_dir: Path    # no test for the bug (the agent must write it)
    issue_file: Path
    test_id: str


def _make_repo(repo: Path, tests: str) -> None:
    remove_dir(repo)
    repo.mkdir(parents=True)
    files = {
        "textkit.py": _TEXTKIT,
        "tests/test_text.py": tests,
        "requirements.txt": "# no runtime dependencies\n",
    }
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    run_git("init", "-q", "-b", "main", cwd=repo)
    run_git("add", "-A", cwd=repo)
    run_git(
        "-c", "user.name=demo", "-c", "user.email=demo@example.com",
        "commit", "-q", "-m", "initial", cwd=repo,
    )


def make_demo(root: Path) -> Demo:
    root.mkdir(parents=True, exist_ok=True)
    repo, blind = root / "textkit", root / "textkit-blind"
    _make_repo(repo, _TESTS_BLIND + _EXTRA_TEST)
    _make_repo(blind, _TESTS_BLIND)
    issue = root / "ISSUE.md"
    issue.write_text(_ISSUE, encoding="utf-8", newline="\n")
    return Demo(repo, blind, issue, DEMO_TEST_ID)