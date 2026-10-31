from pathlib import Path


def safe_resolve(root: Path, rel: str) -> Path | None:
    """Resolve rel under root. None if it escapes root (.., absolute, symlink) or hits .git."""
    root = root.resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        return None
    if ".git" in p.relative_to(root).parts:
        return None
    return p