from pathlib import Path


def is_test_path(path: str) -> bool:
    """Whether path belongs to the test side of a repository."""
    parts = path.split("/")
    name = parts[-1]
    return (
        "tests" in parts[:-1]
        or "test" in parts[:-1]
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def safe_resolve(root: Path, rel: str) -> Path | None:
    """Resolve rel under root. None if it escapes root (.., absolute, symlink) or hits .git."""
    root = root.resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        return None
    if ".git" in p.relative_to(root).parts:
        return None
    return p
