from pathlib import Path

import pytest

from repodoctor.tools.repo import run_git


@pytest.fixture
def make_git_repo(tmp_path):
    def _make(files: dict[str, str | bytes], name: str = "repo") -> Path:
        root = tmp_path / name
        root.mkdir()
        run_git("init", "-q", "-b", "main", cwd=root)
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                p.write_bytes(content)
            else:
                p.write_text(content, encoding="utf-8", newline="\n")
        run_git("add", "-A", cwd=root)
        run_git(
            "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=root
        )
        return root

    return _make